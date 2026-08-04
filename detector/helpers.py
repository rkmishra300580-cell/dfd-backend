"""
helpers.py — v5.2
Shared helper functions used across all modality  pipelines.

New in v5.2:
  - classify_dominant(): three-class decision engine producing dominant
    label + confidence score (x% REAL / x% AI_GENERATED / x% DEEPFAKE)
  - filter_indicators(): strips indicators that contradict the dominant class
  - verdict_text_v2(): legally-safe language tied to dominant class

Original functions unchanged:
  file_metadata, detect_faces, extract_fake_score,
  threat_from_score, verdict_text, apply_graph_style
"""
import os
import hashlib
import mimetypes
from datetime import datetime

import cv2
import matplotlib.pyplot as plt


# ── Classification thresholds (tune without code changes) ─────────────────────
# Synthetic = either deepfake or AI-generated signal is above this.
# 45.0 (raised from 35.0): prevents borderline sdxl-detector false positives
# on real photographs from triggering AI_GENERATED classification.
SYNTHETIC_THRESHOLD  = 45.0
# Editing detected: when edit_composite >= this AND neither the deepfake
# nor AI-generation track already crossed effective_threshold, the image
# classifies as DEEPFAKE (manipulation_type='EDITED') rather than REAL.
# Per product decision: any detected manipulation - AI-driven or manual/
# Photoshop - is DEEPFAKE, not a REAL variant. (Previously this produced a
# 'REAL (Edited)' sub-label; see classify_dominant() for the full change.)
# exif_edit=45 (Photoshop tag) alone → 45*0.70=31.5 → stays REAL (correct)
# exif_edit=45 + manip=30 → 31.5+9=40.5 → DEEPFAKE (Edited)
# exif_edit=70 (stripped+ICC) alone → 49 → DEEPFAKE (Edited)
EDITED_THRESHOLD     = 40.0
# When synthetic: deepfake wins if deepfake_composite >= this
DEEPFAKE_THRESHOLD   = 50.0
# When synthetic: AI_GENERATED wins if ai_gen_composite >= this
AI_GEN_THRESHOLD     = 45.0
# DL AI model floor: replaced 2026-06-29. The narrow check below
# (`dl_ai >= 85.0`) missed a real production case where dl_ai_generated=84.4 -
# 0.6 points short of the threshold - and got zero floor protection as a
# result. Replaced with a general floor applied to BOTH composites, based on
# whichever component is actually strongest, not a fixed score-specific check.
#
# Factor tuned 2026-06-29 via sensitivity sweep against two real cases from a
# Hive comparison: a genuine deepfake (DL=72.2%, correctly stayed DEEPFAKE at
# every floor tested) and a real photo the DL models misread (dl_ai=67.2%,
# falsely became AI_GENERATED at floor>=0.70 given AI_GEN_THRESHOLD=45). 0.65
# is the lowest value where the deepfake case's score is unaffected (plateaus
# at 0.85 and below — the floor stops being the binding constraint past that
# point) while the false-positive case correctly returns to REAL. Do not
# raise this back toward 0.85 without re-running both cases against the
# REAL classify_dominant() function and its actual thresholds (not an
# approximation) — an earlier sweep using an assumed 50-point threshold
# instead of the real 45-point AI_GEN_THRESHOLD wrongly suggested 0.70.
DL_DEEPFAKE_FLOOR = 0.80   # prithivMLmods — validated, face-specific, high trust
DL_AI_FLOOR       = 0.60   # sdxl-detector ensemble — unvalidated, lower trust
# NEW 29 Jul 2026: vehicle_damage_analysis's own heuristics (ELA/shadow/
# texture/boundary/insurance-metadata) are the most directly relevant signal
# for vehicle-bucket images, same role dl_deepfake plays for faces - but
# unlike the DL scores, nothing floored it against dilution by weak,
# unrelated components (dl_ai in particular - see the 29 Jul false-negative
# where vehicle_score=60.9 legitimately elevated but ai_gen_composite only
# reached 29.5 because dl_ai=27.1 dragged it down at 0.45 weight). This
# constant is UNVALIDATED - chosen to match DL_AI_FLOOR's conservative value
# as a reasoned default, not fit to any labeled case. It does NOT flip that
# specific case to correctly-classified (60.9*0.60=36.5, still under
# AI_GEN_THRESHOLD=45.0) - deliberately not tuned to do so, since picking a
# value that would is fitting one example, not calibrating. The real fix is
# running this against evaluate.py on a real labeled set once one exists.
VEHICLE_FLOOR     = 0.60   # vehicle_damage_analysis — unvalidated, needs evaluate.py calibration
# When EITHER ensemble member alone reports confidence at/above this, treat
# it as near-conclusive (see the ceiling applied in _compute_image_composites
# below) rather than letting the ensemble MEAN dilute it. Only fires on the
# raw max score, not the averaged dl_ai used everywhere else - see that
# ceiling's comment for the real case and the known trade-off.
DL_AI_CONCLUSIVE_THRESHOLD = 90.0


# ── Per-modality composite score functions ────────────────────────────────────
# Each returns (deepfake_composite, ai_gen_composite) on a 0-100 scale.
# classify_dominant() dispatches to the right one by payload['file_type'],
# then applies the SAME shared decision logic / risk_level / legacy mapping
# / verdict text regardless of modality - only the composite math differs.

def _compute_image_composites(stage: dict, has_face: bool = True, has_vehicle: bool = None) -> tuple:
    """
    Each composite uses per-model floors (DL_DEEPFAKE_FLOOR / DL_AI_FLOOR).
    strongest_single_component) - a general dominant-signal floor so one
    strong, specific finding can't get diluted below its own significance by
    weaker, unrelated corroborating signals. Applies to BOTH tracks, in
    BOTH branches (face/no-face, vehicle/no-vehicle) - whichever component
    is strongest sets the floor, regardless of which one it happens to be.
    This replaces an earlier version that only floored ai_gen_composite, and
    only when dl_ai >= 85 specifically - a real case with dl_ai=84.4 fell
    just short of that threshold and got no protection at all.

    BUG FIX (29 Jul 2026): has_face used to be silently overwritten two lines
    into this function (`has_face = face is not None`), making the has_face
    PARAMETER dead code regardless of what any caller passed in. Found via a
    live false-negative on a confirmed AI-generated vehicle-damage photo,
    where a Haar false-positive ("face" detected on a wheel/headlight) leaked
    straight into deepfake_composite's weighting formula even though the
    router had already correctly classified the image as VEHICLE - because
    classify_dominant()'s attempt to pass the router's decision in here had
    no effect; this function threw the passed value away immediately.
    has_face/has_vehicle are now genuinely caller-controlled. has_vehicle
    defaults to re-deriving from stage-key presence only when the caller
    doesn't specify it (legacy/non-routed callers) - not silently overridden
    once specified.
    """
    dl_deepfake  = float(stage.get('deep_learning',   0) or 0)
    dl_ai        = float(stage.get('dl_ai_generated', 0) or 0)
    # Raw max of the two AI-gen ensemble members (not the averaged dl_ai
    # above). Falls back to dl_ai itself if a payload predates this field.
    dl_ai_ensemble_max = stage.get('dl_ai_ensemble_max')
    dl_ai_ensemble_max = float(dl_ai_ensemble_max) if dl_ai_ensemble_max is not None else dl_ai
    freq         = float(stage.get('frequency',       0) or 0)
    manip        = float(stage.get('manipulation',    0) or 0)
    face         = stage.get('face_forensics')    # None when no face
    vehicle      = stage.get('vehicle_damage')    # None when face present
    exif_ai      = float(stage.get('exif_ai_score',   0) or 0)
    # True when exif_ai reached its value via the no-EXIF-corroboration block
    # borrowing strength from dl_ai_generated/frequency, rather than from
    # intrinsic EXIF evidence (AI-tool tag, noise contradiction). See gating
    # below at the EXIF-conclusive ceiling for why this distinction matters.
    exif_ai_corroborated = bool(stage.get('exif_ai_corroborated', False))
    exif_edit    = float(stage.get('exif_edit_score',  0) or 0)
    exif_real    = float(stage.get('exif_real_score',  0) or 0)

    if has_vehicle is None:
        has_vehicle = vehicle is not None   # fallback only - legacy/non-routed callers

    EXIF_CONCLUSIVE_AI   = 70.0
    EXIF_CONCLUSIVE_REAL = 65.0

    if has_face:
        # Weights raised from (0.55/0.30/0.15) → (0.70/0.20/0.10).
        # Rationale: the DL model (prithivMLmods/Deep-Fake-Detector-v2-Model)
        # is purpose-trained for face deepfakes and is the authoritative signal.
        # The Haar face score is a region-quality heuristic, not a deepfake
        # detector — over-weighting it was diluting a 76.9% DL hit to 59.4%.
        # Raising DL weight also reduces false-positive risk on real portraits:
        # when DL correctly scores low (~20%), the composite drops vs before
        # even if the Haar score is high (e.g. sharp/symmetric face region).
        face_val = float(face) if face is not None else 0.0
        deepfake_components = [dl_deepfake, face_val]
        deepfake_composite  = dl_deepfake * 0.70 + face_val * 0.10 + manip * 0.20
    else:
        deepfake_components = [manip, dl_deepfake]
        deepfake_composite  = manip * 0.70 + dl_deepfake * 0.30
    deepfake_composite = max(deepfake_composite, dl_deepfake * DL_DEEPFAKE_FLOOR)

    if has_vehicle:
        # dl_ai raised 0.30→0.45: sdxl-detector is the primary signal for
        # non-face images; at 0.30 a 98% score was being drowned out by
        # lower-scoring forensic sub-components.
        vehicle_val = float(vehicle) if vehicle is not None else 0.0
        ai_gen_components = [dl_ai, freq, vehicle_val, exif_ai]
        ai_gen_composite  = dl_ai * 0.45 + freq * 0.20 + vehicle_val * 0.20 + exif_ai * 0.15
    else:
        ai_gen_components = [dl_ai, freq, exif_ai]
        ai_gen_composite  = dl_ai * 0.40 + freq * 0.30 + exif_ai * 0.30
    ai_gen_composite = max(ai_gen_composite, dl_ai * DL_AI_FLOOR)
    if has_vehicle:
        ai_gen_composite = max(ai_gen_composite, vehicle_val * VEHICLE_FLOOR)

    # ── Conclusive-evidence boosts ──────────────────────────────────────────
    # Three separate mechanisms below each say "if ONE signal is extremely
    # confident, trust it — don't let a weighted blend with weaker co-signals
    # dilute it away." They were added at different times for different real
    # cases (ensemble dilution, EXIF-only evidence, clean-noise AI faces).
    # Structural review found they'd been given INCONSISTENT protection
    # against the same failure mode: a real photo that also happens to carry
    # strong genuine-camera EXIF should never be pushed toward synthetic by
    # any of these, but only the first one (added later, after a stress test
    # caught it) actually checked for that. All three now share the exact
    # same gate: `exif_real < EXIF_CONCLUSIVE_REAL`. This is deliberately a
    # hard cutoff (skip the boost entirely), not a proportional dampening —
    # the proportional suppression a few lines below is a separate, weaker
    # safety net for cases that fall under this cutoff.

    # 1. DL AI-Gen ensemble-max ceiling (see full rationale where this was
    #    first introduced): either ensemble member alone reporting >=90%
    #    confidence is treated as near-conclusive.
    if dl_ai_ensemble_max >= DL_AI_CONCLUSIVE_THRESHOLD and exif_real < EXIF_CONCLUSIVE_REAL:
        ai_gen_composite = max(ai_gen_composite, dl_ai_ensemble_max)

    # 2. EXIF conclusive-AI ceiling. Restricted to non-face images (see
    #    rationale below) and to intrinsic (non-corroborated) EXIF evidence,
    #    to avoid double-counting dl_ai/frequency through the EXIF channel.
    #    On face images the deepfake DL model is the primary signal; allowing
    #    a 70% EXIF score to floor ai_gen_composite at 63 can flip
    #    DEEPFAKE->AI_GENERATED even when the face model scored 76.9% higher
    #    on the correct track. EXIF can't distinguish face-swap from
    #    AI-generated; the face pipeline can.
    if exif_ai >= EXIF_CONCLUSIVE_AI and not has_face and not exif_ai_corroborated \
            and exif_real < EXIF_CONCLUSIVE_REAL:
        ai_gen_composite = max(ai_gen_composite, exif_ai * 0.90)

    # 3. H6: standalone low-noise signal on face images. Real camera photos
    #    with faces almost always show noise_residual_std > 3; AI-generated
    #    faces are characteristically clean (std < 2). Fires independently
    #    of EXIF state - catches cases where EXIF was stripped entirely.
    #    Same gate as the two mechanisms above: a real photo taken in good
    #    light can legitimately have low sensor noise too, so this must not
    #    fire when strong real-camera EXIF already contradicts it - this was
    #    the one of the three that structural review found WITHOUT this gate.
    noise_std = float(stage.get('noise_residual_std', 999) or 999)
    if has_face and noise_std < 3.0 and exif_real < EXIF_CONCLUSIVE_REAL:
        noise_contribution = (3.0 - noise_std) / 3.0 * 20.0  # max 20pts at std=0
        ai_gen_composite = min(100.0, ai_gen_composite + noise_contribution)

    # ── Real-camera-evidence suppression (proportional, applies to everything
    # above) ─────────────────────────────────────────────────────────────────
    # Placed AFTER all three conclusive-evidence boosts so it can never be
    # bypassed by a later additive adjustment - every boost above either gets
    # gated out entirely (exif_real >= 65) or, if it fired (exif_real < 65),
    # still passes through this proportional dampening below if exif_real
    # is elevated but under the hard cutoff.
    if exif_real >= EXIF_CONCLUSIVE_REAL:
        suppression_factor = 1.0 - (exif_real - EXIF_CONCLUSIVE_REAL) / 100.0 * 0.6
        suppression_factor = max(suppression_factor, 0.35)
        deepfake_composite  *= suppression_factor
        ai_gen_composite    *= suppression_factor

    # edit_composite: editing software tag (0.70) + forensic manipulation (0.30).
    # A colour-grade alone won't fire ELA/PRNU, so exif_edit carries it solo.
    edit_composite = float(min(max(exif_edit * 0.70 + manip * 0.30, 0), 100))

    return deepfake_composite, ai_gen_composite, edit_composite


def _compute_video_composites(stage: dict) -> tuple:
    """
    AI-generation track: FFT consistency, temporal stillness, and frame-level
    ELA are all general synthesis/editing signals that don't require a face.
    Same general dominant-signal floor as the image composites.
    Deepfake track: face-count consistency is the only currently-computed
    identity-specific signal. It's None (not 0) in stage_scores when no
    faces ever appeared in the video - treated as 0 contribution here,
    same "no face, no identity to fake" principle as the image pipeline.
    """
    fft   = float(stage.get('video_fft_suspicion',      0) or 0)
    temp  = float(stage.get('video_temporal_suspicion', 0) or 0)
    ela   = float(stage.get('video_ela_suspicion',      0) or 0)
    face  = stage.get('video_face_suspicion')   # None when no faces in any sampled frame

    ai_gen_composite   = fft * 0.45 + temp * 0.35 + ela * 0.20
    ai_gen_composite   = max(ai_gen_composite, max(fft, temp, ela) * DL_AI_FLOOR)
    deepfake_composite = float(face) if face is not None else 0.0
    return deepfake_composite, ai_gen_composite, 0.0  # edit_composite N/A for video


def _compute_audio_composites(stage: dict) -> tuple:
    """
    AI-generation track: all five current features (MFCC, spectral flatness,
    phase irregularity, ZCR, bandwidth) detect synthetic/TTS audio in general.
    Deepfake track: hardcoded 0. None of the current features verify whether
    a voice matches a specific target identity - that needs speaker/voice
    verification, which isn't implemented. Don't change this to a computed
    heuristic without actually building that capability.
    """
    ai_gen_composite   = float(stage.get('audio_ai_generated', stage.get('audio_forensics', 0)) or 0)
    deepfake_composite = 0.0
    return deepfake_composite, ai_gen_composite, 0.0  # edit_composite N/A for audio


def _compute_document_composites(stage: dict) -> tuple:
    """
    AI-generation track: the existing AI-text-detector + entropy + uniformity
    blend already IS the AI-generation signal.
    Deepfake track: hardcoded 0. A document has no identity to impersonate -
    "deepfake" isn't a meaningful concept for text.
    """
    ai_gen_composite   = float(stage.get('document_ai_generated', stage.get('document_forensics', 0)) or 0)
    deepfake_composite = 0.0
    return deepfake_composite, ai_gen_composite, 0.0  # edit_composite N/A for documents


def classify_dominant(payload: dict) -> dict:
    """
    Three-class hierarchical decision engine.

    Reads scores already written to payload by the image pipeline:
        stage_scores.deep_learning      → deepfake DL score
        stage_scores.dl_ai_generated    → AI-generation DL score
        stage_scores.frequency          → forensic AI-gen signal
        stage_scores.manipulation       → forensic manipulation signal
        stage_scores.face_forensics     → face deepfake signal (None if no face)
        stage_scores.vehicle_damage     → vehicle AI-gen signal (None if face present)

    Returns a dict of new fields to merge into payload.
    No existing payload fields are modified.

    Decision logic (two levels):
      Level 1: REAL vs SYNTHETIC
        synthetic_score = max(deepfake_composite, ai_gen_composite)
        if synthetic_score < SYNTHETIC_THRESHOLD → REAL

      Level 2: which type?
        if deepfake_composite >= DEEPFAKE_THRESHOLD → DEEPFAKE
        elif ai_gen_composite >= AI_GEN_THRESHOLD   → AI_GENERATED
        else                                         → REAL (low confidence synthetic)

    Dominant score = confidence in the winning class (0-100).
    This is what the frontend shows as the headline number.
    """
    stage          = payload.get('stage_scores', {})
    file_type      = payload.get('file_type', 'IMAGE')
    content_bucket = payload.get('content_bucket')  # set by router.py for IMAGE only, since 28 Jul 2026

    if file_type == 'IMAGE':
        # BUG FIX (29 Jul 2026, found from a live false-negative on a
        # confirmed AI-generated vehicle-damage photo): this used to read
        # raw Haar output (`stage.get('face_forensics') is not None`)
        # directly, completely independent of the router's content_bucket
        # decision made in image_pipeline.py. Haar false-positived a "face"
        # on the car's wheel/headlight (a known Haar failure mode - dark
        # circular/textured regions), which made this function compute
        # deepfake_composite using the FACE-weighted formula
        # (dl_deepfake*0.70 + face*0.10 + manip*0.20) instead of the
        # correct VEHICLE-content formula (manip*0.70 + dl_deepfake*0.30) -
        # even though the router had already correctly classified the
        # image as VEHICLE. Two independent fusion pathways
        # (dl_detector's adaptive fusion in image_pipeline.py, and this
        # function) each had their own face/vehicle determination, and
        # only one of them got wired to the router in the original routing
        # fix. has_face here is now derived from content_bucket when the
        # router has run, falling back to raw Haar output only for older
        # payloads that predate routing (should not occur in practice
        # post-deploy, kept defensively).
        has_face_for_composite = (
            (content_bucket == 'FACE') if content_bucket is not None
            else (stage.get('face_forensics') is not None)
        )
        has_vehicle_for_composite = (
            (content_bucket == 'VEHICLE') if content_bucket is not None
            else None  # let _compute_image_composites fall back to stage-key presence
        )
        deepfake_composite, ai_gen_composite, edit_composite = _compute_image_composites(
            stage, has_face=has_face_for_composite, has_vehicle=has_vehicle_for_composite
        )
    elif file_type == 'VIDEO'    : deepfake_composite, ai_gen_composite, edit_composite = _compute_video_composites(stage)
    elif file_type == 'AUDIO'    : deepfake_composite, ai_gen_composite, edit_composite = _compute_audio_composites(stage)
    elif file_type == 'DOCUMENT' : deepfake_composite, ai_gen_composite, edit_composite = _compute_document_composites(stage)
    else                         : deepfake_composite, ai_gen_composite, edit_composite = 0.0, 0.0, 0.0
    ai_gen_composite   = float(min(max(ai_gen_composite,   0), 100))
    edit_composite     = float(min(max(edit_composite,      0), 100))
    synthetic_score    = max(deepfake_composite, ai_gen_composite)  # edit is sub-REAL, not synthetic
    real_score         = float(max(0, 100 - synthetic_score))

    # ── Effective threshold: raised when camera EXIF signals present ──────────
    # exif_real > 0 means some camera metadata exists. A real photo with even
    # partial camera EXIF needs a stronger synthetic signal to be overridden.
    # exif_real=15 raises threshold 45→49.5; exif_real=65 raises it to 64.5.
    # Only applies to IMAGE modality where exif_real is computed.
    exif_real_for_threshold = float(stage.get('exif_real_score', 0) or 0)
    effective_threshold = SYNTHETIC_THRESHOLD + exif_real_for_threshold * 0.30

    has_face = stage.get('face_forensics') is not None

    # ── Two-level decision ────────────────────────────────────────────────────
    if synthetic_score < effective_threshold:
        # Neither the deepfake nor AI-generation track crossed the bar - but
        # editing evidence (Photoshop/Lightroom tag + forensic corroboration)
        # is a SEPARATE, third signal checked here.
        #
        # TAXONOMY CHANGE (28 Jul 2026): reversed the previous "all
        # manipulation = DEEPFAKE" merge. Per explicit product decision,
        # manual/Photoshop editing (no AI/deep learning involved) is once
        # again classified as REAL (with editing_detected=True and its own
        # 'REAL (Edited)' badge), separate from DEEPFAKE which is now
        # reserved for AI-driven manipulation only (face-swap or AI-based
        # editing of a real photo). manipulation_type still distinguishes
        # 'EDITED' from 'FACE_SWAP'/'MANIPULATION' for anyone reading the
        # report who needs to know which (e.g. an insurance investigator
        # cares whether a claim photo was Photoshopped vs. AI face-swapped —
        # they now show DIFFERENT top-level badges, matching that need).
        #
        # This matches a pre-existing frontend contract (app/page.js
        # CLF_CONFIG.REAL_EDITED / editing_detected promotion logic) that
        # was already built and shipped but unreachable, because this
        # branch previously always returned classification='DEEPFAKE' —
        # the REAL+editing_detected combination the frontend was waiting
        # for could never actually occur. It's reachable now.
        editing_detected = (edit_composite >= EDITED_THRESHOLD)
        if editing_detected:
            classification    = 'REAL'
            dominant_score    = edit_composite
            manipulation_type = 'EDITED'
            dominant_label    = f'{dominant_score:.0f}% REAL (Edited)'
        else:
            classification    = 'REAL'
            dominant_score    = real_score
            manipulation_type = None
            dominant_label    = f'{dominant_score:.0f}% REAL'

    elif deepfake_composite >= DEEPFAKE_THRESHOLD and deepfake_composite >= ai_gen_composite:
        classification    = 'DEEPFAKE'
        dominant_score    = deepfake_composite
        manipulation_type = 'FACE_SWAP' if has_face else 'MANIPULATION'
        dominant_label    = f'{dominant_score:.0f}% DEEPFAKE'
        editing_detected  = False

    elif ai_gen_composite >= AI_GEN_THRESHOLD:
        classification    = 'AI_GENERATED'
        dominant_score    = ai_gen_composite
        manipulation_type = None
        dominant_label    = f'{dominant_score:.0f}% AI GENERATED'
        editing_detected  = False

    else:
        # Synthetic signal present but neither track is dominant enough
        # → lean toward whichever composite is higher, flag as low confidence
        if deepfake_composite >= ai_gen_composite:
            classification   = 'DEEPFAKE'
            dominant_score   = deepfake_composite
            manipulation_type = 'FACE_SWAP' if has_face else 'MANIPULATION'
        else:
            classification   = 'AI_GENERATED'
            dominant_score   = ai_gen_composite
            manipulation_type = None
        dominant_label   = f'{dominant_score:.0f}% {classification.replace("_", " ")} (low confidence)'
        editing_detected = False

    # ── Risk level ────────────────────────────────────────────────────────────
    # NOTE: 'REAL' no longer implies zero risk on its own — a REAL image with
    # editing_detected=True (Photoshop/manual manipulation, e.g. an altered
    # insurance claim photo) still needs a risk tier scaled to how strong the
    # editing evidence is, same tiering as DEEPFAKE/AI_GENERATED. Only a
    # genuinely clean REAL (no editing found) is unconditionally LOW.
    #
    # BUG FIX (31 Jul 2026, found from a live case): the old bands had
    # `dominant_score < 50 -> LOW`, which overlapped with AI_GEN_THRESHOLD
    # (45.0) - a confirmed AI_GENERATED classification at e.g. 46.3% showed
    # "LOW risk" despite being a positive finding, because 46.3 < 50. Any
    # classification that reaches this branch (AI_GENERATED, DEEPFAKE, or
    # REAL+edited) already crossed its own suspicion threshold (40-50 range
    # depending on class) to get classified that way in the first place -
    # LOW is never the right tier for a confirmed positive finding,
    # regardless of exactly how far past its threshold the score landed.
    if classification == 'REAL' and not editing_detected:
        risk_level = 'LOW'
    elif dominant_score < 65:
        risk_level = 'MODERATE'
    elif dominant_score < 80:
        risk_level = 'HIGH'
    else:
        risk_level = 'CRITICAL'

    # ── Legacy final_score mapping ─────────────────────────────────────────────
    # IMPORTANT: final_score is now set EQUAL to dominant_score, not a separate
    # remapped value. It was previously computed via a different linear formula
    # (e.g. `40 + ai_gen_composite * 0.3`), which meant the frontend's headline
    # number (reading final_score) and the verdict paragraph's embedded score
    # (reading dominant_score) showed two different numbers for the same
    # result - e.g. "59% FAKE" headline next to "(score 64/100)" in the verdict
    # text. There is no old frontend left that needs the old remapped range;
    # making these the same value by construction is what actually fixes that,
    # not just narrowing the gap between two still-separate formulas.
    # Same editing_detected carve-out as risk_level above: a REAL+edited image
    # should report its edit-evidence strength (dominant_score = edit_composite),
    # not the low synthetic_score that this branch exists for pure-REAL images.
    if classification == 'REAL' and not editing_detected:
        legacy_score = round(synthetic_score, 1)
    else:
        legacy_score = dominant_score

    # verdict_text_v2() now returns a dict (4 Aug 2026 - see its own
    # docstring) instead of a bare string, so it's called once here and
    # unpacked into three payload fields rather than assigned directly to
    # 'verdict'.
    _verdict_fields = verdict_text_v2(classification, dominant_score,
                                       deepfake_composite, ai_gen_composite,
                                       file_type=file_type,
                                       editing_detected=editing_detected,
                                       manipulation_type=manipulation_type,
                                       risk_level=risk_level)

    return {
        # ── New fields (dominant classification) ──────────────────────────────
        'classification':      classification,
        'dominant_score':      round(dominant_score, 1),
        'dominant_label':      dominant_label,      # e.g. "78% DEEPFAKE" or "64% DEEPFAKE (Edited)"
        'real_score':          round(real_score, 1),
        'ai_generated_score':  round(ai_gen_composite, 1),
        'deepfake_score':      round(deepfake_composite, 1),
        'edited_score':        round(edit_composite, 1),   # editing signal strength
        'editing_detected':    editing_detected,  # True → classification is REAL via the editing track specifically ("REAL (Edited)"), not a synthetic classification
        'manipulation_type':   manipulation_type, # 'FACE_SWAP' | 'MANIPULATION' | 'EDITED' | None — 'EDITED' now pairs with classification='REAL'; 'FACE_SWAP'/'MANIPULATION' pair with classification='DEEPFAKE'
        'risk_level':          risk_level,
        # ── Updated legacy fields ─────────────────────────────────────────────
        'final_score':         round(legacy_score, 1),
        'threat_level':        threat_from_score(legacy_score),
        'verdict':             _verdict_fields['verdict'],
        # NEW (4 Aug 2026): additive fields for the frontend to render a
        # distinct callout/action element instead of just the paragraph.
        # Both may be None - that's expected (e.g. plain REAL has no
        # highlight sentence and, at LOW risk, no recommended action).
        'verdict_highlight':   _verdict_fields['verdict_highlight'],
        'recommended_action':  _verdict_fields['recommended_action'],
    }



def filter_indicators(indicators: list, classification: str,
                      has_human_face: bool = True,
                      manipulation_type: str = None) -> list:
    """
    Return only indicators that support the dominant classification
    AND are appropriate for the detected content type (face vs vehicle).

    Two-pass filter:
      Pass 1 — Content type gate (face vs vehicle)
        If has_human_face=True:  remove ALL [Vehicle] and [Insurance] indicators
        If has_human_face=False: remove ALL [Face] indicators
        This is unconditional — vehicle indicators must never appear on face
        images regardless of classification, and vice versa.

      Pass 2 — Classification gate
        REAL         → keep only indicators that support authenticity;
                       EXCEPTION: if manipulation_type=='EDITED' (REAL with
                       editing_detected=True, i.e. "REAL (Edited)"), keep
                       the [EXIF]/[Manipulation] editing-evidence tags
                       (ELA/PRNU/copy-move/editing-software) instead of
                       returning an empty list — that evidence is exactly
                       what earned the "(Edited)" qualifier and must not be
                       silently dropped.
        DEEPFAKE     → manipulation_type distinguishes which evidence to keep:
                       'FACE_SWAP'/'MANIPULATION' → [Face], [Manipulation],
                       [DL] identity-manipulation signals.
                       'EDITED' → same tags as the REAL/'EDITED' case above,
                       kept here only as a defensive fallback — as of
                       28 Jul 2026 this combination (classification=
                       DEEPFAKE + manipulation_type=EDITED) should not
                       normally occur, since editing now classifies as REAL.
                       See classify_dominant()'s 28 Jul 2026 taxonomy note.
        AI_GENERATED → keep [Frequency], [EXIF] AI-gen signals, [Vehicle]
                       (already gated out for face images in Pass 1).
                       Drop [Face] deepfake signals.

    The full unfiltered list is in payload['all_indicators'] for debugging.
    """
    if not indicators:
        return []

    # ── Pass 1: Content type gate ─────────────────────────────────────────────
    VEHICLE_TAGS = ['[Vehicle]', '[Insurance]', 'vehicle', 'insurance',
                    'damage', 'panel', 'accident', 'inpainted']
    FACE_TAGS    = ['[Face]', 'resolution mismatch', 'face edge',
                    'blur smoothing on face', 'boundary blending',
                    'multi-scale energy', 'resampling artifacts in face']

    def _matches(indicator: str, tags: list) -> bool:
        ind_lower = indicator.lower()
        return any(t.lower() in ind_lower for t in tags)

    if has_human_face:
        # Face image: strip all vehicle/insurance indicators unconditionally
        indicators = [i for i in indicators if not _matches(i, VEHICLE_TAGS)]
    else:
        # Vehicle/object image: strip face-specific indicators
        indicators = [i for i in indicators if not _matches(i, FACE_TAGS)]

    # ── Pass 2: Classification gate ───────────────────────────────────────────
    DEEPFAKE_TAGS = [
        '[Face]', '[Manipulation]', '[DL]',
        'boundary', 'blur', 'resolution mismatch', 'copy-move',
        'patch inconsistency', 'resampling', 'ELA', 'PRNU',
        'noise inconsistency',                     # PRNU = manipulation signal
        '[EXIF] professional editing',              # editing sw = manipulation
        '[EXIF] ai generator',                     # AI generator tag
        '[EXIF] no metadata',                      # conclusive AI signal
        '[EXIF] metadata dimensions',               # thumbnail mismatch
    ]
    AI_GEN_TAGS = [
        '[Frequency]', '[EXIF]',
        'sensor-noise', 'spectral entropy', 'edge density',
        'high-frequency', 'AI generator', 'AI-generated',
        'no metadata', 'minimal metadata', 'metadata stripped',
        'flattened', 'weak',                      # frequency signals
    ]

    # Modality-specific tag sets, checked SEPARATELY from the image-oriented
    # ones above. Without this, e.g. video's "[Video] High ELA score on
    # sampled frames" would get swept into DEEPFAKE_TAGS purely because it
    # contains the substring "ELA" - which means something different in the
    # video context (general frame-editing signal, not face-specific) than
    # it does for images (PRNU/manipulation, face-adjacent). Gating by the
    # bracket prefix first prevents any cross-modality word collision.
    VIDEO_DEEPFAKE_TAGS  = ['consistent face count']
    VIDEO_AI_GEN_TAGS    = ['consistent fft', 'consistent frame fft', 'temporal motion',
                            'ela score on sampled frames', 'ela on sampled frames']
    AUDIO_AI_GEN_TAGS    = ['mfcc variance', 'spectral flatness', 'spectral bandwidth',
                            'low mfcc', 'high spectral', 'narrow spectral']
    DOCUMENT_AI_GEN_TAGS = ['ai-text detector', 'unusually uniform sentence']

    def _modality(indicator: str) -> str:
        if indicator.startswith('[Video]'):    return 'VIDEO'
        if indicator.startswith('[Audio]'):    return 'AUDIO'
        if indicator.startswith('[Document]'): return 'DOCUMENT'
        return 'IMAGE'

    # Shared with the DEEPFAKE/'EDITED' defensive fallback below — the actual
    # evidence tags that support an editing finding regardless of which
    # top-level classification carries it.
    EDITED_TAGS = ['[EXIF]', '[Manipulation]', 'editing software',
                   'ELA', 'PRNU', 'copy-move', 'patch', 'metadata']

    if classification == 'REAL':
        if manipulation_type == 'EDITED':
            # REAL (Edited): keep the evidence that actually drove this —
            # EXIF editing-software tag + manipulation forensics (ELA/PRNU/
            # copy-move) — same tag set as the old DEEPFAKE/'EDITED' path
            # used, just reachable from the REAL branch now that editing
            # classifies as REAL. Getting this wrong here would silently
            # reproduce the exact "REAL (Edited) always shows zero
            # indicators" bug this file's own history already had once,
            # in the opposite direction — see 28 Jul 2026 taxonomy note in
            # classify_dominant().
            return [i for i in indicators if _matches(i, EDITED_TAGS)]
        # Genuinely clean REAL: indicator list is typically empty after
        # Pass 1 anyway. Any remaining indicators are borderline — suppress
        # them all so the user isn't confused by low-confidence noise on a
        # REAL result.
        return []

    elif classification == 'DEEPFAKE':
        result = []
        for i in indicators:
            mod = _modality(i)
            if manipulation_type == 'EDITED':
                # Defensive fallback only — as of 28 Jul 2026, editing
                # evidence classifies as REAL (see above), not DEEPFAKE, so
                # this combination should not normally occur. Kept in case
                # a future scoring change reintroduces it, so indicators
                # don't silently disappear again if it does.
                keep = _matches(i, EDITED_TAGS)
            elif mod == 'VIDEO':
                keep = _matches(i, VIDEO_DEEPFAKE_TAGS)
            elif mod in ('AUDIO', 'DOCUMENT'):
                keep = False  # no deepfake-track indicators exist for these
            else:
                keep = _matches(i, DEEPFAKE_TAGS)
            if keep: result.append(i)
        return result

    elif classification == 'AI_GENERATED':
        # Also keep manipulation signals that are relevant to AI generation
        # (copy-move, ELA on the whole image — not face-specific ones).
        ai_gen_manip = ['[Manipulation] High ELA', '[Manipulation] Regional ELA',
                        '[Manipulation] [Metadata]', 'copy-move', 'patch inconsistency']
        result = []
        for i in indicators:
            mod = _modality(i)
            if   mod == 'VIDEO'    : keep = _matches(i, VIDEO_AI_GEN_TAGS)
            elif mod == 'AUDIO'    : keep = _matches(i, AUDIO_AI_GEN_TAGS)
            elif mod == 'DOCUMENT' : keep = _matches(i, DOCUMENT_AI_GEN_TAGS)
            else                    : keep = _matches(i, AI_GEN_TAGS) or _matches(i, ai_gen_manip)
            if keep: result.append(i)
        return result

    # Unknown / fallback — return what survived Pass 1
    return indicators


def _recommended_action_line(risk_level: str) -> str:
    """
    Risk-scaled next-action sentence, appended to the verdict for any
    result that isn't a clean REAL (risk_level == 'LOW'). Kept in the same
    cautious, non-definitive register as the rest of verdict_text_v2 -
    "recommended", never "must" or "do not approve" - this is guidance for
    a human reviewer, not an automated denial.
    """
    return {
        'MODERATE': 'Recommended next step: a manual review before this result is relied on for a claim decision.',
        'HIGH':     'Recommended next step: review by a forensic analyst before this result is used in a claim decision.',
        'CRITICAL': 'Recommended next step: this result should not be used to approve or deny a claim without manual forensic review first.',
    }.get(risk_level)


def verdict_text_v2(classification: str, dominant_score: float,
                    deepfake_score: float, ai_gen_score: float,
                    file_type: str = 'IMAGE',
                    editing_detected: bool = False,
                    manipulation_type: str = None,
                    risk_level: str = None) -> dict:
    """
    Legally safe verdict language tied to dominant classification.
    Avoids definitive statements. Phrasing is modality-aware - "captured by
    a real camera" doesn't make sense for a document, and image-style
    "face-swapping" language doesn't fit a cloned voice.
    manipulation_type=='EDITED' produces distinct 'this was edited, not
    face-swapped/AI-generated' language within the REAL branch - as of
    28 Jul 2026 the top-level classification is REAL (not DEEPFAKE) for
    manual/non-AI editing; a reader still needs to know editing evidence
    was found even though the badge says REAL.

    CHANGED (4 Aug 2026): returns a dict instead of a bare string -
    {'verdict', 'verdict_highlight', 'recommended_action'} - additive, not
    a breaking rename: 'verdict' still holds the exact same full paragraph
    text as before (now with the recommended_action sentence appended when
    risk_level warrants one, so PDF and any plain-text consumer gets it for
    free with zero PDF-layout changes needed). The two new keys are for
    consumers (the frontend) that want to render a distinct emphasized
    callout instead of just a paragraph:
      verdict_highlight   - the single most reassuring/important sentence,
                             pulled out verbatim from 'verdict' (currently
                             only set for REAL+Edited - the sentence
                             clarifying "real source, conventionally
                             altered, not AI" is the one point most worth
                             a user's immediate attention). None otherwise.
      recommended_action  - the same risk-scaled action sentence that's
                             already appended into 'verdict', exposed
                             separately so the frontend can style it as its
                             own "next step" element if desired. None for
                             risk_level == 'LOW' (nothing to recommend).
    """
    score_str = f'{dominant_score:.0f}/100'
    action = _recommended_action_line(risk_level)

    DEEPFAKE_PHRASING = {
        'IMAGE':    'possible face-swapping, identity substitution, or targeted synthetic alteration of authentic source media',
        'VIDEO':    'possible face-swapping or identity substitution within the video',
        'AUDIO':    'possible voice cloning or identity-specific audio manipulation',
        'DOCUMENT': 'targeted alteration of the document content',
    }
    AI_GEN_PHRASING = {
        'IMAGE':    'an AI image generator rather than captured by a real camera',
        'VIDEO':    'an AI video generation tool, or contains synthetically generated frames, rather than captured by a real camera',
        'AUDIO':    'speech synthesis or voice generation (TTS) rather than a genuine recording',
        'DOCUMENT': 'an AI text generation tool rather than written by a human author',
    }

    def _finish(text: str, highlight: str = None) -> dict:
        full = f'{text} {action}' if action else text
        return {'verdict': full, 'verdict_highlight': highlight, 'recommended_action': action}

    if classification == 'REAL':
        if editing_detected and manipulation_type == 'EDITED':
            highlight = (
                'This differs from face-swapping or AI-generated content: the underlying '
                'media appears to originate from a real source that was subsequently '
                'altered by conventional (non-AI) means, rather than being synthetically '
                'generated or having an identity substituted.'
            )
            return _finish(
                f'Forensic analysis detected evidence of editing or post-processing '
                f'(score {score_str}) - for example colour grading, retouching, cloning, '
                f'or other manipulation-software traces. {highlight} This assessment is '
                f'based on automated forensic analysis and should be verified by a '
                f'qualified analyst before being used as evidence.',
                highlight=highlight,
            )
        return _finish(
            'No significant indicators of synthetic manipulation were detected. '
            'Content appears consistent with authentic, unedited media under '
            'current forensic analysis.'
        )
    elif classification == 'DEEPFAKE':
        if manipulation_type == 'EDITED':
            # Defensive fallback only - as of 28 Jul 2026 this combination
            # should not normally occur (editing now classifies as REAL,
            # handled above). Kept so verdict text doesn't silently
            # mismatch the indicator list if this state is ever reached.
            return _finish(
                f'Forensic analysis detected evidence of editing or post-processing '
                f'(score {score_str}) - for example colour grading, retouching, cloning, '
                f'or other manipulation-software traces. This differs from face-swapping '
                f'or AI-generated content: the underlying media appears to originate from '
                f'a real source that was subsequently altered, rather than being '
                f'synthetically generated or having an identity substituted. This '
                f'assessment is based on automated forensic analysis and should be '
                f'verified by a qualified analyst before being used as evidence.'
            )
        phrase = DEEPFAKE_PHRASING.get(file_type, DEEPFAKE_PHRASING['IMAGE'])
        return _finish(
            f'Multiple indicators associated with deepfake manipulation were detected '
            f'(score {score_str}). Analysis suggests {phrase}. '
            f'This assessment is based on automated forensic analysis and should be '
            f'verified by a qualified analyst before being used as evidence.'
        )
    elif classification == 'AI_GENERATED':
        phrase = AI_GEN_PHRASING.get(file_type, AI_GEN_PHRASING['IMAGE'])
        return _finish(
            f'Multiple indicators consistent with AI-generated content were detected '
            f'(score {score_str}). Analysis suggests this content may have been produced by '
            f'{phrase}. '
            f'This assessment is based on automated forensic analysis and should be '
            f'verified by a qualified analyst before being used as evidence.'
        )
    else:
        return _finish(
            'Forensic analysis produced insufficient or contradictory evidence to '
            'make a reliable classification. Manual review by a qualified analyst '
            'is recommended.'
        )



# ── Original helpers — unchanged ──────────────────────────────────────────────

def file_metadata(filepath):
    return {
        'name'      : os.path.basename(filepath),
        'size_bytes': os.path.getsize(filepath),
        'mime'      : mimetypes.guess_type(filepath)[0] or 'unknown',
        'md5'       : hashlib.md5(open(filepath, 'rb').read()).hexdigest(),
        'sha256'    : hashlib.sha256(open(filepath, 'rb').read()).hexdigest(),
        'analyzed_at': datetime.now().isoformat(),
    }


def detect_faces(rgb_array, min_confidence=0.4):
    """
    OpenCV Haar cascade face detector.
    Mediapipe removed — OOM risk on Render.
    min_confidence kept for API compat but unused by Haar.
    """
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    cc   = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = cc.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
    return [(int(x), int(y), int(fw), int(fh)) for (x, y, fw, fh) in faces] \
        if len(faces) > 0 else []


FAKE_LABEL_VARIANTS = [
    'Deepfake', 'deepfake', 'DEEPFAKE', 'Fake', 'fake', 'FAKE',
    'artificial', 'Artificial', 'manipulated', 'Manipulated',
    'synthetic', 'Synthetic', 'generated', 'Generated',
]

# NOTE: prithivMLmods/Deep-Fake-Detector-v2-Model confirmed inverted labels.
# Validated at 92.9% accuracy with this fix. DO NOT REMOVE.
def extract_fake_score(label_map):
    for key in FAKE_LABEL_VARIANTS:
        if key in label_map:
            return (1 - label_map[key]) * 100, key
    real_variants = {'real', 'realism', 'authentic', 'genuine', 'original'}
    for label, score in label_map.items():
        if label.lower() not in real_variants:
            return (1 - score) * 100, label
    return 0.0, 'unknown'


def threat_from_score(score: float) -> str:
    if   score >= 90: return 'CRITICAL'
    elif score >= 75: return 'HIGH'
    elif score >= 50: return 'MODERATE'
    elif score >= 25: return 'LOW'
    else:             return 'MINIMAL'


def verdict_text(score: float) -> str:
    """Legacy single-score verdict — kept for non-image modalities."""
    if   score >= 75: return 'Multiple independent forensic and DL indicators strongly suggest synthetic or AI-generated content.'
    elif score >= 50: return 'Several suspicious patterns detected. Content may be manipulated or AI-generated.'
    elif score >= 25: return 'Weak indicators of possible synthetic manipulation detected.'
    else:             return 'Content appears largely authentic under current forensic analysis.'


GRAPH_STYLE = {
    'figure.facecolor': '#0d1117',
    'axes.facecolor'  : '#161b22',
    'axes.edgecolor'  : '#30363d',
    'axes.labelcolor' : '#c9d1d9',
    'text.color'      : '#c9d1d9',
    'xtick.color'     : '#8b949e',
    'ytick.color'     : '#8b949e',
    'grid.color'      : '#21262d',
    'grid.linestyle'  : '--',
}

def apply_graph_style():
    plt.rcParams.update(GRAPH_STYLE)
