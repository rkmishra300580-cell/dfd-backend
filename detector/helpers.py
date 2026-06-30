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
# Editing detected: when exif_edit_composite >= this AND classification resolves
# to REAL, the label becomes 'REAL (Edited)'. Sub-class of REAL — not synthetic.
# exif_edit=45 (Photoshop tag) alone → 45*0.70=31.5 → stays REAL (correct)
# exif_edit=45 + manip=30 → 31.5+9=40.5 → REAL (Edited)
# exif_edit=70 (stripped+ICC) alone → 49 → REAL (Edited)
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
COMPOSITE_FLOOR_FACTOR = 0.65


# ── Per-modality composite score functions ────────────────────────────────────
# Each returns (deepfake_composite, ai_gen_composite) on a 0-100 scale.
# classify_dominant() dispatches to the right one by payload['file_type'],
# then applies the SAME shared decision logic / risk_level / legacy mapping
# / verdict text regardless of modality - only the composite math differs.

def _compute_image_composites(stage: dict, has_face: bool = True) -> tuple:
    """
    Each composite uses max(weighted_average, COMPOSITE_FLOOR_FACTOR *
    strongest_single_component) - a general dominant-signal floor so one
    strong, specific finding can't get diluted below its own significance by
    weaker, unrelated corroborating signals. Applies to BOTH tracks, in
    BOTH branches (face/no-face, vehicle/no-vehicle) - whichever component
    is strongest sets the floor, regardless of which one it happens to be.
    This replaces an earlier version that only floored ai_gen_composite, and
    only when dl_ai >= 85 specifically - a real case with dl_ai=84.4 fell
    just short of that threshold and got no protection at all.
    """
    dl_deepfake  = float(stage.get('deep_learning',   0) or 0)
    dl_ai        = float(stage.get('dl_ai_generated', 0) or 0)
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

    has_face    = face    is not None
    has_vehicle = vehicle is not None

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
        deepfake_components = [dl_deepfake, float(face)]
        deepfake_composite  = dl_deepfake * 0.70 + float(face) * 0.20 + manip * 0.10
    else:
        deepfake_components = [manip, dl_deepfake]
        deepfake_composite  = manip * 0.70 + dl_deepfake * 0.30
    deepfake_composite = max(deepfake_composite, max(deepfake_components) * COMPOSITE_FLOOR_FACTOR)

    if has_vehicle:
        # dl_ai raised 0.30→0.45: sdxl-detector is the primary signal for
        # non-face images; at 0.30 a 98% score was being drowned out by
        # lower-scoring forensic sub-components.
        ai_gen_components = [dl_ai, freq, float(vehicle), exif_ai]
        ai_gen_composite  = dl_ai * 0.45 + freq * 0.20 + float(vehicle) * 0.20 + exif_ai * 0.15
    else:
        ai_gen_components = [dl_ai, freq, exif_ai]
        ai_gen_composite  = dl_ai * 0.40 + freq * 0.30 + exif_ai * 0.30
    ai_gen_composite = max(ai_gen_composite, max(ai_gen_components) * COMPOSITE_FLOOR_FACTOR)

    # EXIF conclusive-AI ceiling only applies when there is no human face,
    # AND when exif_ai reflects intrinsic EXIF evidence (not corroboration-
    # derived). Two reasons:
    # 1. (existing) On face images the deepfake DL model is the primary
    #    signal; allowing a 70% EXIF score to floor ai_gen_composite at 63
    #    can flip DEEPFAKE->AI_GENERATED even when the face-specific model
    #    scored 76.9% — wrong trade-off. EXIF can't distinguish face-swap
    #    from AI-generated; the face pipeline can.
    # 2. (new) When exif_ai reached 70 via the no-EXIF-corroboration block
    #    (image_pipeline.py), it got there by borrowing strength from
    #    dl_ai_generated or frequency — signals already counted directly in
    #    the weighted ai_gen_composite average above, and already protected
    #    by COMPOSITE_FLOOR_FACTOR if they're dominant. Applying the EXIF
    #    ceiling on top double-counts that same signal a second time via a
    #    different path. Real case: dl_ai=84.4 corroborates exif_ai to 70;
    #    weighted composite is already 62.4 (dl_ai counted once, correctly);
    #    the ceiling then pushed it to 63.0 by re-counting dl_ai's influence
    #    through the EXIF channel. Skipping the ceiling here doesn't lose
    #    protection — COMPOSITE_FLOOR_FACTOR (0.65 x strongest component)
    #    already guarantees ai_gen_composite >= dl_ai*0.65 = 54.9 in this case.
    if exif_ai >= EXIF_CONCLUSIVE_AI and not has_face and not exif_ai_corroborated:
        ai_gen_composite = max(ai_gen_composite, exif_ai * 0.90)

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
    ai_gen_composite   = max(ai_gen_composite, max(fft, temp, ela) * COMPOSITE_FLOOR_FACTOR)
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
    stage     = payload.get('stage_scores', {})
    file_type = payload.get('file_type', 'IMAGE')

    if   file_type == 'IMAGE'    : deepfake_composite, ai_gen_composite, edit_composite = _compute_image_composites(stage, has_face=stage.get('face_forensics') is not None)
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

    # ── Two-level decision ────────────────────────────────────────────────────
    if synthetic_score < effective_threshold:
        classification   = 'REAL'
        dominant_score   = real_score
        # Sub-classification: was the image edited even if not synthetically generated?
        # editing_detected is set here and carried in the return dict.
        # The frontend uses it to display 'REAL (Edited)' instead of plain 'REAL'.
        # Threshold is conservative: requires editing software tag + some forensic
        # corroboration, so a bare Photoshop tag on an untouched export stays 'REAL'.
        editing_detected = (edit_composite >= EDITED_THRESHOLD)
        dominant_label   = f'{dominant_score:.0f}% REAL (Edited)' if editing_detected else f'{dominant_score:.0f}% REAL'

    elif deepfake_composite >= DEEPFAKE_THRESHOLD and deepfake_composite >= ai_gen_composite:
        classification   = 'DEEPFAKE'
        dominant_score   = deepfake_composite
        dominant_label   = f'{dominant_score:.0f}% DEEPFAKE'
        editing_detected = False

    elif ai_gen_composite >= AI_GEN_THRESHOLD:
        classification   = 'AI_GENERATED'
        dominant_score   = ai_gen_composite
        dominant_label   = f'{dominant_score:.0f}% AI GENERATED'
        editing_detected = False

    else:
        # Synthetic signal present but neither track is dominant enough
        # → lean toward whichever composite is higher, flag as low confidence
        if deepfake_composite >= ai_gen_composite:
            classification = 'DEEPFAKE'
            dominant_score = deepfake_composite
        else:
            classification = 'AI_GENERATED'
            dominant_score = ai_gen_composite
        dominant_label   = f'{dominant_score:.0f}% {classification.replace("_", " ")} (low confidence)'
        editing_detected = False

    # ── Risk level ────────────────────────────────────────────────────────────
    if classification == 'REAL':
        risk_level = 'LOW'
    elif dominant_score < 50:
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
    legacy_score = dominant_score

    return {
        # ── New fields (dominant classification) ──────────────────────────────
        'classification':      classification,
        'dominant_score':      round(dominant_score, 1),
        'dominant_label':      dominant_label,      # e.g. "78% DEEPFAKE" or "92% REAL (Edited)"
        'real_score':          round(real_score, 1),
        'ai_generated_score':  round(ai_gen_composite, 1),
        'deepfake_score':      round(deepfake_composite, 1),
        'edited_score':        round(edit_composite, 1),   # NEW: editing signal strength
        'editing_detected':    editing_detected,           # NEW: True → show 'REAL (Edited)'
        'risk_level':          risk_level,
        # ── Updated legacy fields ─────────────────────────────────────────────
        'final_score':         round(legacy_score, 1),
        'threat_level':        threat_from_score(legacy_score),
        'verdict':             verdict_text_v2(classification, dominant_score,
                                               deepfake_composite, ai_gen_composite,
                                               file_type=file_type,
                                               editing_detected=editing_detected),
    }


def filter_indicators(indicators: list, classification: str,
                      has_human_face: bool = True) -> list:
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
        REAL         → keep only indicators that support authenticity
                       (in practice almost none fire for REAL verdicts)
        DEEPFAKE     → keep [Face], [Manipulation], [DL], [EXIF] manipulation
                       signals. Drop pure AI-generation frequency signals.
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

    if classification == 'REAL':
        # For REAL verdicts the indicator list is typically empty after Pass 1.
        # Any remaining indicators are borderline — suppress them all so the
        # user isn't confused by low-confidence noise on a REAL result.
        return []

    elif classification == 'DEEPFAKE':
        result = []
        for i in indicators:
            mod = _modality(i)
            if   mod == 'VIDEO'                 : keep = _matches(i, VIDEO_DEEPFAKE_TAGS)
            elif mod in ('AUDIO', 'DOCUMENT')    : keep = False  # no deepfake-track indicators exist for these
            else                                  : keep = _matches(i, DEEPFAKE_TAGS)
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

    elif classification == 'REAL':
        # When editing_detected=True the classification is still 'REAL' — keep
        # EXIF editing indicators and manipulation forensics. Drop AI-gen and
        # deepfake-specific signals so they don't confuse the user.
        EDITED_TAGS = ['[EXIF]', '[Manipulation]', 'editing software',
                       'ELA', 'PRNU', 'copy-move', 'patch', 'metadata']
        # If no editing indicators exist (plain REAL), this returns [] per
        # the existing REAL branch above — this branch is only reached for
        # REAL (Edited) where indicators survived Pass 1.
        result = [i for i in indicators if _matches(i, EDITED_TAGS)]
        return result

    # Unknown / fallback — return what survived Pass 1
    return indicators


def verdict_text_v2(classification: str, dominant_score: float,
                    deepfake_score: float, ai_gen_score: float,
                    file_type: str = 'IMAGE',
                    editing_detected: bool = False) -> str:
    """
    Legally safe verdict language tied to dominant classification.
    Avoids definitive statements. Phrasing is modality-aware - "captured by
    a real camera" doesn't make sense for a document, and image-style
    "face-swapping" language doesn't fit a cloned voice.
    editing_detected=True produces 'REAL (Edited)' language when classification=REAL.
    """
    score_str = f'{dominant_score:.0f}/100'

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

    if classification == 'REAL':
        if editing_detected:
            return (
                'Content appears to originate from a genuine source, but forensic '
                'analysis detected evidence of professional editing or post-processing. '
                'The underlying image is not assessed as AI-generated or deepfake. '
                'Edits may include colour grading, retouching, cropping, or other '
                'common photo-editing workflows. This does not indicate fabrication '
                'or synthetic generation.'
            )
        return (
            'No significant indicators of synthetic manipulation were detected. '
            'Content appears consistent with authentic, unedited media under '
            'current forensic analysis.'
        )
    elif classification == 'DEEPFAKE':
        phrase = DEEPFAKE_PHRASING.get(file_type, DEEPFAKE_PHRASING['IMAGE'])
        return (
            f'Multiple indicators associated with deepfake manipulation were detected '
            f'(score {score_str}). Analysis suggests {phrase}. '
            f'This assessment is based on automated forensic analysis and should be '
            f'verified by a qualified analyst before being used as evidence.'
        )
    elif classification == 'AI_GENERATED':
        phrase = AI_GEN_PHRASING.get(file_type, AI_GEN_PHRASING['IMAGE'])
        return (
            f'Multiple indicators consistent with AI-generated content were detected '
            f'(score {score_str}). Analysis suggests this content may have been produced by '
            f'{phrase}. '
            f'This assessment is based on automated forensic analysis and should be '
            f'verified by a qualified analyst before being used as evidence.'
        )
    else:
        return (
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
