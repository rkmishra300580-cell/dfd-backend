"""
detector/router.py
Content-type router for the image pipeline.

Runs BEFORE any type-specific forensic stage. Classifies an image into
exactly one of three buckets:
  FACE     -> face_pipeline.py     (face_forensic_analysis, unchanged logic)
  VEHICLE  -> vehicle_pipeline.py  (vehicle_damage_analysis, unchanged logic)
  OTHER    -> other_pipeline.py    (content-agnostic forensics only, NEW)

Method: CLIP zero-shot classification (openai/clip-vit-base-patch32).
Chosen over training a custom classifier because it requires no labeled
training data, is free/open, and fits the existing "load -> infer -> fully
release" memory pattern already used by every other HF model in this
codebase (see image_pipeline.py header comment: models are never resident
at the same time on the 2GB Render instance). This function follows that
exact pattern and must not be changed to a persistent singleton without
re-confirming the memory tradeoff first.

Decision policy (confirmed with project owner, 28 Jul 2026):
  If neither FACE nor VEHICLE clears CONFIDENCE_THRESHOLD, OR the top two
  buckets are within MARGIN_THRESHOLD of each other -> default to OTHER.
  This is the deliberately conservative choice: OTHER runs only the
  content-agnostic forensic stages (frequency / ELA / copy-move / PRNU /
  metadata / DL AI-gen ensemble), so an uncertain classification can never
  trigger fabricated vehicle-damage or face-forensic indicators on the
  wrong subject. Safest failure mode = fewest false positives, matching
  the owner's standing instruction to prioritize correctness over coverage.

KNOWN LIMITATION (not yet resolved, flag before relying on this in prod):
  This module has been syntax/logic-checked only (py_compile clean, no
  import errors in a mock environment). The huggingface.co domain used to
  download CLIP weights is not reachable from the development sandbox
  (only pypi/npm/github registries are allowlisted there), so actual
  inference against real images has NOT been run outside of Render. The
  real test is the next live deploy - do not treat this docstring as
  evidence the router works correctly on real photos.
"""
import gc
import torch
from transformers import pipeline as hf_pipeline

CONFIDENCE_THRESHOLD = 0.45   # winning bucket must clear this score
MARGIN_THRESHOLD      = 0.12  # winning bucket must beat runner-up by at least this

# CLIP candidate prompts, mapped to the 3 output buckets. Two OTHER-flavoured
# prompts are used (documents/screenshots and general scenes) so their
# scores can be summed - this avoids CLIP splitting "not a face, not a
# vehicle" probability mass across two labels and under-counting OTHER.
_LABEL_TO_BUCKET = {
    "a photo of a human face or person":              "FACE",
    "a photo of a vehicle, car, or vehicle damage":    "VEHICLE",
    "a document, receipt, form, or screenshot":        "OTHER",
    "a general photo of a scene, object, or animal":   "OTHER",
}
_CANDIDATE_LABELS = list(_LABEL_TO_BUCKET.keys())


def _decide_bucket(raw_scores: dict) -> dict:
    """
    Pure decision logic, deliberately separated from the CLIP model call so
    it can be unit-tested without loading torch/transformers or reaching
    huggingface.co. Takes CLIP's raw per-label scores and applies the
    confidence/margin policy described in the module docstring.

    Args:
        raw_scores: {label: score} for every label in _CANDIDATE_LABELS.
                    Scores need not sum to 1 (defensive - CLIP zero-shot
                    scores are already softmax-normalised across candidates,
                    but this function does not assume that).

    Returns: see classify_content_type() docstring for the return shape.
    """
    bucket_scores = {"FACE": 0.0, "VEHICLE": 0.0, "OTHER": 0.0}
    for label, score in raw_scores.items():
        bucket_scores[_LABEL_TO_BUCKET[label]] += score

    ranked = sorted(bucket_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_bucket, top_score = ranked[0]
    second_bucket, second_score = ranked[1]

    if top_bucket == "OTHER":
        return {
            "bucket": "OTHER",
            "confidence": top_score,
            "raw_scores": raw_scores,
            "reason": f"OTHER scored highest ({top_score:.2f})",
        }

    if top_score < CONFIDENCE_THRESHOLD:
        return {
            "bucket": "OTHER",
            "confidence": top_score,
            "raw_scores": raw_scores,
            "reason": (f"Top bucket {top_bucket} below confidence threshold "
                       f"({top_score:.2f} < {CONFIDENCE_THRESHOLD}) -> defaulting to OTHER"),
        }

    if (top_score - second_score) < MARGIN_THRESHOLD:
        return {
            "bucket": "OTHER",
            "confidence": top_score,
            "raw_scores": raw_scores,
            "reason": (f"{top_bucket} ({top_score:.2f}) too close to {second_bucket} "
                       f"({second_score:.2f}); margin < {MARGIN_THRESHOLD} -> defaulting to OTHER"),
        }

    return {
        "bucket": top_bucket,
        "confidence": top_score,
        "raw_scores": raw_scores,
        "reason": f"{top_bucket} scored highest ({top_score:.2f}) with sufficient margin",
    }


def classify_content_type(pil_image) -> dict:
    """
    Route a single image into FACE / VEHICLE / OTHER.

    Args:
        pil_image: a PIL.Image instance (any mode; converted to RGB here).

    Returns:
        {
            "bucket":     "FACE" | "VEHICLE" | "OTHER",
            "confidence": float,               # winning bucket's summed CLIP score
            "raw_scores": {label: score, ...}, # full CLIP output, for logging
            "reason":     str,                 # human-readable justification
        }
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = hf_pipeline(
        'zero-shot-image-classification',
        model='openai/clip-vit-base-patch32',
        device=0 if device == 'cuda' else -1
    )
    try:
        raw = classifier(pil_image.convert('RGB'), candidate_labels=_CANDIDATE_LABELS)
    finally:
        del classifier
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_scores = {r['label']: float(r['score']) for r in raw}
    return _decide_bucket(raw_scores)
