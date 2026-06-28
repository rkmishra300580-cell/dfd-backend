"""
helpers.py — v5.2
Shared helper functions used across all modality pipelines.

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
# Synthetic = either deepfake or AI-generated signal is above this
SYNTHETIC_THRESHOLD  = 35.0
# When synthetic: deepfake wins if dl_deepfake_score >= this
DEEPFAKE_THRESHOLD   = 50.0
# When synthetic: AI_GENERATED wins if dl_ai_score >= this
AI_GEN_THRESHOLD     = 45.0


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
    stage = payload.get('stage_scores', {})

    dl_deepfake  = float(stage.get('deep_learning',   0) or 0)
    dl_ai        = float(stage.get('dl_ai_generated', 0) or 0)
    freq         = float(stage.get('frequency',       0) or 0)
    manip        = float(stage.get('manipulation',    0) or 0)
    face         = stage.get('face_forensics')    # None when no face
    vehicle      = stage.get('vehicle_damage')    # None when face present
    exif_ai      = float(stage.get('exif_ai_score',   0) or 0)
    exif_real    = float(stage.get('exif_real_score',  0) or 0)

    has_face    = face    is not None
    has_vehicle = vehicle is not None

    # ── EXIF override rules ───────────────────────────────────────────────────
    # EXIF is the highest-certainty signal we have when it fires conclusively.
    # An AI generator software tag (exif_ai >= 95) or zero JPEG EXIF
    # (exif_ai >= 70) should dominate the result — they are near-definitive.
    # A fully-populated camera EXIF with optics (exif_real >= 65) is a strong
    # REAL signal that should suppress synthetic classification unless DL model
    # or manipulation analysis strongly contradicts it.
    #
    # Implementation: EXIF scores act as floor/ceiling on composites,
    # not just additive weights. This prevents a mediocre DL score from
    # overriding a definitive EXIF finding.
    EXIF_CONCLUSIVE_AI   = 70.0   # at or above this → treat exif_ai as floor on ai_gen_composite
    EXIF_CONCLUSIVE_REAL = 65.0   # at or above this → treat exif_real as ceiling suppressor

    # ── Composite deepfake score ──────────────────────────────────────────────
    if has_face:
        deepfake_composite = (
            dl_deepfake  * 0.55 +
            float(face)  * 0.30 +
            manip        * 0.15
        )
    else:
        deepfake_composite = manip * 0.70 + dl_deepfake * 0.30

    # ── Composite AI-generation score ─────────────────────────────────────────
    # EXIF now has explicit weight — previously it was diluted through
    # manipulation's metadata sub-score at 0.08 * 0.30 = 0.024 effective weight.
    # Now exif_ai carries 0.30 weight in face mode, 0.25 in vehicle mode.
    if has_vehicle:
        ai_gen_composite = (
            dl_ai          * 0.30 +
            freq           * 0.20 +
            float(vehicle) * 0.25 +
            exif_ai        * 0.25
        )
    else:
        ai_gen_composite = (
            dl_ai   * 0.40 +
            freq    * 0.30 +
            exif_ai * 0.30
        )

    # ── Apply EXIF floor/ceiling ──────────────────────────────────────────────
    # If EXIF conclusively signals AI generation, ensure ai_gen_composite
    # is at least as high as the EXIF score (it cannot be drowned out by
    # other low-scoring components).
    if exif_ai >= EXIF_CONCLUSIVE_AI:
        ai_gen_composite = max(ai_gen_composite, exif_ai * 0.90)

    # If EXIF conclusively confirms a real camera, suppress synthetic composites.
    # We don't zero them out — DL or manipulation might still be right —
    # but we cap how high they can go relative to the EXIF real signal.
    if exif_real >= EXIF_CONCLUSIVE_REAL:
        suppression_factor = 1.0 - (exif_real - EXIF_CONCLUSIVE_REAL) / 100.0 * 0.6
        suppression_factor = max(suppression_factor, 0.35)  # never suppress below 35%
        deepfake_composite  *= suppression_factor
        ai_gen_composite    *= suppression_factor

    deepfake_composite = float(min(max(deepfake_composite, 0), 100))
    ai_gen_composite   = float(min(max(ai_gen_composite,   0), 100))
    synthetic_score    = max(deepfake_composite, ai_gen_composite)
    real_score         = float(max(0, 100 - synthetic_score))

    # ── Two-level decision ────────────────────────────────────────────────────
    if synthetic_score < SYNTHETIC_THRESHOLD:
        classification   = 'REAL'
        dominant_score   = real_score
        dominant_label   = f'{dominant_score:.0f}% REAL'

    elif deepfake_composite >= DEEPFAKE_THRESHOLD and deepfake_composite >= ai_gen_composite:
        classification   = 'DEEPFAKE'
        dominant_score   = deepfake_composite
        dominant_label   = f'{dominant_score:.0f}% DEEPFAKE'

    elif ai_gen_composite >= AI_GEN_THRESHOLD:
        classification   = 'AI_GENERATED'
        dominant_score   = ai_gen_composite
        dominant_label   = f'{dominant_score:.0f}% AI GENERATED'

    else:
        # Synthetic signal present but neither track is dominant enough
        # → lean toward whichever composite is higher, flag as low confidence
        if deepfake_composite >= ai_gen_composite:
            classification = 'DEEPFAKE'
            dominant_score = deepfake_composite
        else:
            classification = 'AI_GENERATED'
            dominant_score = ai_gen_composite
        dominant_label = f'{dominant_score:.0f}% {classification.replace("_", " ")} (low confidence)'

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
        'dominant_label':      dominant_label,      # e.g. "78% DEEPFAKE"
        'real_score':          round(real_score, 1),
        'ai_generated_score':  round(ai_gen_composite, 1),
        'deepfake_score':      round(deepfake_composite, 1),
        'risk_level':          risk_level,
        # ── Updated legacy fields ─────────────────────────────────────────────
        'final_score':         round(legacy_score, 1),
        'threat_level':        threat_from_score(legacy_score),
        'verdict':             verdict_text_v2(classification, dominant_score,
                                               deepfake_composite, ai_gen_composite),
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

    if classification == 'REAL':
        # For REAL verdicts the indicator list is typically empty after Pass 1.
        # Any remaining indicators are borderline — suppress them all so the
        # user isn't confused by low-confidence noise on a REAL result.
        return []

    elif classification == 'DEEPFAKE':
        return [i for i in indicators if _matches(i, DEEPFAKE_TAGS)]

    elif classification == 'AI_GENERATED':
        # For AI_GENERATED: keep frequency/EXIF signals.
        # Also keep manipulation signals that are relevant to AI generation
        # (copy-move, ELA on the whole image — not face-specific ones).
        ai_gen_manip = ['[Manipulation] High ELA', '[Manipulation] Regional ELA',
                        '[Manipulation] [Metadata]', 'copy-move', 'patch inconsistency']
        return [
            i for i in indicators
            if _matches(i, AI_GEN_TAGS) or _matches(i, ai_gen_manip)
        ]

    # Unknown / fallback — return what survived Pass 1
    return indicators


def verdict_text_v2(classification: str, dominant_score: float,
                    deepfake_score: float, ai_gen_score: float) -> str:
    """
    Legally safe verdict language tied to dominant classification.
    Avoids definitive statements. Uses hedge language appropriate
    for forensic reports.
    """
    score_str = f'{dominant_score:.0f}/100'

    if classification == 'REAL':
        return (
            'No significant indicators of synthetic manipulation were detected. '
            'Content appears consistent with authentic, unedited media under '
            'current forensic analysis.'
        )
    elif classification == 'DEEPFAKE':
        return (
            f'Multiple indicators associated with deepfake manipulation were detected '
            f'(score {score_str}). Analysis suggests possible face-swapping, identity '
            f'substitution, or targeted synthetic alteration of authentic source media. '
            f'This assessment is based on automated forensic analysis and should be '
            f'verified by a qualified analyst before being used as evidence.'
        )
    elif classification == 'AI_GENERATED':
        return (
            f'Multiple indicators consistent with AI-generated content were detected '
            f'(score {score_str}). Analysis suggests this content may have been produced '
            f'by an AI image generator rather than captured by a real camera. '
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
