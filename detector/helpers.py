"""
Shared helper functions used across all modality pipelines:
file metadata, multi-tier face detection, DL-label score extraction
(with the inverted-label fix applied), scoring/threat-level mapping,
and shared graph styling.

Logic carried over unchanged from the validated Colab prototype (Cell 5).
"""
import os
import hashlib
import mimetypes
from datetime import datetime

import cv2
import matplotlib.pyplot as plt


def file_metadata(filepath):
    return {
        'name'     : os.path.basename(filepath),
        'size_bytes': os.path.getsize(filepath),
        'mime'     : mimetypes.guess_type(filepath)[0] or 'unknown',
        'md5'      : hashlib.md5(open(filepath, 'rb').read()).hexdigest(),
        'sha256'   : hashlib.sha256(open(filepath, 'rb').read()).hexdigest(),
        'analyzed_at': datetime.now().isoformat(),
    }


def detect_faces(rgb_array, min_confidence=0.4):
    """
    OpenCV Haar cascade face detector.
    Mediapipe was removed — it initialised TFLite + EGL on every call,
    consuming ~150 MB and triggering OOM kills on Render Starter (512 MB).
    Haar cascade is a plain XML file: negligible memory, no GPU init.
    The min_confidence parameter is kept for API compatibility but unused
    (Haar uses fixed thresholds via scaleFactor/minNeighbors).
    """
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    cc   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cc.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
    return [(int(x), int(y), int(fw), int(fh)) for (x, y, fw, fh) in faces] if len(faces) > 0 else []


FAKE_LABEL_VARIANTS = [
    'Deepfake', 'deepfake', 'DEEPFAKE', 'Fake', 'fake', 'FAKE',
    'artificial', 'Artificial', 'manipulated', 'Manipulated',
    'synthetic', 'Synthetic', 'generated', 'Generated',
]

# NOTE: prithivMLmods/Deep-Fake-Detector-v2-Model has a confirmed inverted
# label mapping in its HuggingFace config (acknowledged by the model author,
# see model discussion #2, Nov 2025: 'It seems the mapping is inverted in
# the HF config'). The pipeline returns {'label': 'Deepfake', 'score': X}
# when the model actually means 'Realism' with confidence X, and vice versa.
# Validated via batch testing on labeled Kaggle data (92.9% accuracy after
# this fix, vs. ~7% before). Re-check this if the upstream model is updated.
def extract_fake_score(label_map):
    for key in FAKE_LABEL_VARIANTS:
        if key in label_map:
            return (1 - label_map[key]) * 100, key
    real_variants = {'real', 'realism', 'authentic', 'genuine', 'original'}
    for label, score in label_map.items():
        if label.lower() not in real_variants:
            return (1 - score) * 100, label
    return 0.0, 'unknown'


def threat_from_score(score):
    if   score >= 90: return 'CRITICAL'
    elif score >= 75: return 'HIGH'
    elif score >= 50: return 'MODERATE'
    elif score >= 25: return 'LOW'
    else:             return 'MINIMAL'


def verdict_text(score):
    if   score >= 75: return 'Multiple independent forensic and DL indicators strongly suggest synthetic or AI-generated content.'
    elif score >= 50: return 'Several suspicious patterns detected. Content may be manipulated or AI-generated.'
    elif score >= 25: return 'Weak indicators of possible synthetic manipulation detected.'
    else:             return 'Content appears largely authentic under current forensic analysis.'


# Dark theme matplotlib style for all graphs
GRAPH_STYLE = {
    'figure.facecolor' : '#0d1117',
    'axes.facecolor'   : '#161b22',
    'axes.edgecolor'   : '#30363d',
    'axes.labelcolor'  : '#c9d1d9',
    'text.color'       : '#c9d1d9',
    'xtick.color'      : '#8b949e',
    'ytick.color'      : '#8b949e',
    'grid.color'       : '#21262d',
    'grid.linestyle'   : '--',
}


def apply_graph_style():
    plt.rcParams.update(GRAPH_STYLE)
