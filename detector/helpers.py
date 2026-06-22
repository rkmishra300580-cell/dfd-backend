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
import mediapipe as mp
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
    """Try MediaPipe tasks API → legacy → OpenCV Haar in order."""
    h, w = rgb_array.shape[:2]

    # Attempt 1: MediaPipe tasks API (>= 0.10)
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core import base_options as mp_base
        import urllib.request
        model_path = '/tmp/blaze_face_short_range.tflite'
        if not os.path.exists(model_path):
            url = ('https://storage.googleapis.com/mediapipe-models/'
                   'face_detector/blaze_face_short_range/float16/1/'
                   'blaze_face_short_range.tflite')
            urllib.request.urlretrieve(url, model_path)
        base_opts   = mp_base.BaseOptions(model_asset_path=model_path)
        detect_opts = mp_vision.FaceDetectorOptions(
            base_options=base_opts, min_detection_confidence=min_confidence)
        detector = mp_vision.FaceDetector.create_from_options(detect_opts)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
        result   = detector.detect(mp_img)
        boxes = []
        for det in result.detections:
            bb = det.bounding_box
            boxes.append((
                max(0, bb.origin_x), max(0, bb.origin_y),
                min(bb.width,  w - bb.origin_x),
                min(bb.height, h - bb.origin_y)
            ))
        return boxes
    except Exception:
        pass

    # Attempt 2: MediaPipe legacy
    try:
        mp_fd = mp.solutions.face_detection
        with mp_fd.FaceDetection(model_selection=1,
                                  min_detection_confidence=min_confidence) as det:
            res = det.process(rgb_array)
        boxes = []
        if res and res.detections:
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                fx = max(0, int(bb.xmin * w)); fy = max(0, int(bb.ymin * h))
                fw = min(int(bb.width * w), w-fx); fh = min(int(bb.height * h), h-fy)
                boxes.append((fx, fy, fw, fh))
        return boxes
    except Exception:
        pass

    # Attempt 3: OpenCV Haar
    gray  = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    cc    = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cc.detectMultiScale(gray, 1.05, 3, minSize=(40, 40))
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
