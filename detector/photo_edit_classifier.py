"""
detector/photo_edit_classifier.py

Domain-specific classifier for FACE/OTHER-bucket images: genuine real
photo vs. real photo carrying local AI editing (inpainting, background
swap, object add/remove, targeted retouch by tools like Photoshop
Generative Fill, etc.) - as opposed to a wholesale AI-generated image.

WHY THIS EXISTS
----------------
Both general-purpose AI-generation detectors used elsewhere in this
pipeline (Organika/sdxl-detector, umm-maybe/AI-image-detector - see
image_pipeline.py's dl_detector()) were confirmed, 14 Aug 2026, to be
validated for a DIFFERENT task than the one Darpan actually needs on
FACE/OTHER content:

  - Organika/sdxl-detector was fine-tuned on Wikimedia-photo vs.
    SDXL-full-regeneration-of-that-photo pairs. Its own model card
    states performance may be lower for generators other than SDXL.
    (https://huggingface.co/Organika/sdxl-detector)
  - umm-maybe/AI-image-detector's own model card states its intended
    scope is artistic images, that it is explicitly NOT a deepfake
    photo detector, and that general computer imagery (webcams,
    screenshots, etc.) may throw it off.
    (https://huggingface.co/umm-maybe/AI-image-detector)

Neither model has ever been validated on "ordinary real photo, locally
edited by an AI tool" - the exact fraud pattern this product exists to
catch (see PROJECT HANDOFF, 11 Aug 2026: "take a real photo, ask a
conversational AI to edit it"). Six real production reports (13-14 Aug
2026: a3c8e0d26e56, b4bdd97c7671, 68d50b30b32c, 3bd5e4b313f7,
ed8f93052f47, 688ec2426020) showed exactly the erratic behavior you'd
expect from two out-of-domain models on this input class: the sdxl
ensemble member swung across nearly the full 0-100 range (2.3 to 97.6,
stdev 41.7) while the umm-maybe member stayed clustered in an
uninformative 36-67 band (stdev 10.1) regardless of ground truth -
consistent with one model guessing wildly and the other one
consistently hedging near uncertain, rather than either one actually
recognizing the content class.

This follows the exact same fix already applied to vehicle-domain
content (see vehicle_ai_gen_classifier.py, 4 Aug 2026): rather than
continuing to tune fusion weights on two signals with a low ceiling for
this task, train a lightweight classifier on your own labeled domain
data, using a frozen CLIP model as a feature extractor (cheap - no GPU
needed, trains in minutes once data exists) and a small logistic-
regression head on top.

STATUS AS OF THIS FILE'S CREATION: UNTRAINED.
No model file exists yet. score_photo_edit_domain() returns None until
train_and_save() (see train_photo_edit_classifier.py for the CLI) has
been run against real labeled data and produced
detector/models/photo_edit_clf.joblib.

DATA NEEDED: two labeled sets, at least MIN_PER_CLASS each -
  - REAL:      genuine, unedited photos (no AI editing of any kind).
  - AI_EDITED: real photos that have had AI editing applied (the six
    reports listed above are a starting point for this class, but more
    are needed - especially covering different edit types: background
    change, object add/remove, face retouch, lighting/style transfer).
Per the project's real-evaluation-before-shipping standard, this MUST
include genuine REAL examples, not just AI_EDITED ones - training or
validating on AI_EDITED-only data would repeat the exact one-sided-
tuning mistake already identified as a root cause elsewhere in this
codebase.

EVERY CALLER MUST TREAT None AS "SIGNAL UNAVAILABLE" AND DEGRADE
GRACEFULLY - same pattern as vehicle_domain_score / dl_ai_score being
None elsewhere in this codebase. None is NOT a score of 0; treating it
as 0 would silently assert "not edited" with no basis.

DEPENDENCY NOTE: needs scikit-learn and joblib, same as
vehicle_ai_gen_classifier.py - confirm both are in requirements.txt
before relying on this in production (vehicle_ai_gen_classifier.py's
own docstring flags this was missing as of its creation; check whether
it was ever added). The sklearn/joblib imports inside
score_photo_edit_domain() are wrapped in try/except ImportError
specifically so a missing dependency degrades to None rather than
crashing every FACE/OTHER-bucket request - that guard is a safety net,
not a substitute for actually adding the dependency.
"""
import os
import json
import gc

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

# Same CLIP checkpoint already used by router.py's content-type router and
# by vehicle_ai_gen_classifier.py - reusing a choice already validated for
# memory footprint on this 2GB Render instance, not introducing a new model
# to load/flush.
_CLIP_MODEL_ID = 'openai/clip-vit-base-patch32'

_MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')
_MODEL_PATH = os.path.join(_MODEL_DIR, 'photo_edit_clf.joblib')
_META_PATH  = os.path.join(_MODEL_DIR, 'photo_edit_clf_meta.json')

_CLASSES = ['AI_EDITED', 'REAL']  # fixed, alphabetical - matches sklearn's default classes_ ordering


def _extract_clip_embedding(pil_image) -> np.ndarray:
    """
    Loads CLIP, extracts one image's embedding, fully releases the model
    before returning - same load-score-flush pattern as every other HF
    model in this codebase (see image_pipeline.py header comment: no two
    models resident at once on the 2GB Render instance). Identical to
    vehicle_ai_gen_classifier.py's version; kept as a separate copy here
    rather than a shared import so this module can be deleted independently
    if the domain-classifier approach doesn't pan out for photo content -
    same reasoning as not sharing state between the two domain classifiers.
    """
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
    model     = CLIPModel.from_pretrained(_CLIP_MODEL_ID).to(device)
    model.eval()
    try:
        inputs = processor(images=pil_image.convert('RGB'), return_tensors='pt').to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        # BUG FIX (26 Aug 2026): same fix as vehicle_ai_gen_classifier.py -
        # get_image_features() is documented to return a raw tensor, but was
        # confirmed to return a wrapped model-output object instead under
        # certain transformers versions ("'BaseModelOutputWithPooling' object
        # has no attribute 'cpu'"). This file has never been run/trained yet,
        # so the bug hadn't triggered here, but the code is identical to
        # vehicle_ai_gen_classifier.py's - fixing proactively rather than
        # waiting to hit the same crash whenever this classifier is first used.
        if hasattr(features, 'cpu'):
            embedding = features.cpu().numpy().flatten()
        elif hasattr(features, 'image_embeds'):
            embedding = features.image_embeds.cpu().numpy().flatten()
        elif hasattr(features, 'pooler_output'):
            embedding = features.pooler_output.cpu().numpy().flatten()
        else:
            raise TypeError(f'Unexpected return type from get_image_features(): {type(features)}')
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return embedding


def score_photo_edit_domain(pil_image):
    """
    Returns a 0-100 float (confidence the image is a REAL photo carrying
    AI editing) if a trained model exists, otherwise None.

    Callers MUST handle None as "signal unavailable" - see module
    docstring. Only raises if a model file exists but is corrupt/
    unreadable (a real error worth surfacing), never for "not trained yet"
    or "dependency missing" - both are expected states right now and
    degrade to None silently by design.
    """
    if not os.path.exists(_MODEL_PATH):
        return None

    try:
        import joblib
    except ImportError:
        return None

    clf = joblib.load(_MODEL_PATH)
    embedding = _extract_clip_embedding(pil_image)
    proba = clf.predict_proba(embedding.reshape(1, -1))[0]
    edited_index = list(clf.classes_).index('AI_EDITED')
    return float(proba[edited_index] * 100)


def train_and_save(real_image_paths: list, ai_edited_image_paths: list,
                    test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    OFFLINE TRAINING - not called during request handling. Run manually via
    train_photo_edit_classifier.py once labeled images exist for BOTH
    classes.

    Returns real held-out evaluation metrics (accuracy, confusion matrix,
    per-class precision/recall) - printed AND returned, never hidden. Per
    project standard: no fix ships on a reasoned guess, only on real
    evaluation results, and that applies to this classifier's own accuracy
    just as much as to any threshold change elsewhere in this codebase.

    Raises ValueError if either class has fewer than MIN_PER_CLASS images -
    training a classifier on fewer than that and deploying it silently
    would be worse than refusing to train at all.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import joblib
    from PIL import Image as PILImage

    MIN_PER_CLASS = 10
    if len(real_image_paths) < MIN_PER_CLASS or len(ai_edited_image_paths) < MIN_PER_CLASS:
        raise ValueError(
            f'Need at least {MIN_PER_CLASS} images per class to train a '
            f'meaningful classifier. Got {len(real_image_paths)} REAL, '
            f'{len(ai_edited_image_paths)} AI_EDITED. Training refused.'
        )

    X, y, failed = [], [], []
    for path in real_image_paths:
        try:
            X.append(_extract_clip_embedding(PILImage.open(path)))
            y.append('REAL')
        except Exception as e:
            failed.append((path, str(e)))
    for path in ai_edited_image_paths:
        try:
            X.append(_extract_clip_embedding(PILImage.open(path)))
            y.append('AI_EDITED')
        except Exception as e:
            failed.append((path, str(e)))

    if failed:
        print(f'WARNING: {len(failed)} images failed to embed and were skipped:')
        for path, err in failed:
            print(f'  {path}: {err}')

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # CALIBRATION FIX applied proactively (28 Aug 2026) - not a theoretical
    # precaution. face_ai_gen_classifier.py's bare LogisticRegression showed
    # a real, reproducible pattern of confidently-wrong predictions across 3
    # consecutive batches (max confidence on a WRONG prediction: 99.67%,
    # 99.93%, 88.7%) before this exact fix was applied there. Applying it
    # here from the start rather than waiting to rediscover the same
    # problem - same reasoning: a bare LogisticRegression's predict_proba()
    # isn't guaranteed to be calibrated, and this file's ai-edited-vs-real
    # task is if anything a HARDER, more overlapping distinction than
    # wholesale-AI-vs-real (see this module's own docstring on the CLIP-
    # embedding architecture's suitability doubts for local-edit detection)
    # - if anything, MORE risk of overconfident wrongness here, not less.
    # method='sigmoid' (Platt scaling), not 'isotonic' - sklearn's own
    # guidance is to prefer sigmoid below ~1,000 samples to avoid the
    # calibration step itself overfitting.
    # cv fold count capped by the smallest class's actual training size,
    # not hardcoded - MIN_PER_CLASS only guarantees 10 images BEFORE the
    # train/test split and BEFORE any embedding failures.
    # score_photo_edit_domain() needs NO changes for this -
    # CalibratedClassifierCV exposes the identical .predict_proba()/
    # .classes_ interface a bare LogisticRegression does.
    base_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    min_class_count = min(np.bincount(np.unique(y_train, return_inverse=True)[1]))
    cv_folds = max(2, min(5, min_class_count))
    clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=cv_folds)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # CALIBRATION CHECK - same as face_ai_gen_classifier.py: reports, for
    # the held-out test set, per-file confidence (distance from 50/50,
    # matching the deployed shape) split by correct vs incorrect
    # predictions. Healthy pattern: incorrect predictions show LOWER
    # confidence than correct ones (unsure when wrong, not confidently
    # wrong). Red flag: incorrect predictions with confidence >= 80%.
    proba_test = clf.predict_proba(X_test)
    ai_edited_col = list(clf.classes_).index('AI_EDITED')
    proba_ai_edited = proba_test[:, ai_edited_col]
    per_file_confidence = np.abs(proba_ai_edited - 0.5) * 200
    correct_mask = (y_pred == y_test)

    calibration = {
        'correct_confidence_mean':   float(per_file_confidence[correct_mask].mean()) if correct_mask.any() else None,
        'correct_confidence_min':    float(per_file_confidence[correct_mask].min()) if correct_mask.any() else None,
        'incorrect_confidence_mean': float(per_file_confidence[~correct_mask].mean()) if (~correct_mask).any() else None,
        'incorrect_confidence_max':  float(per_file_confidence[~correct_mask].max()) if (~correct_mask).any() else None,
        'n_incorrect_high_confidence': int(np.sum((~correct_mask) & (per_file_confidence >= 80))),
    }

    metrics = {
        'accuracy':                float(accuracy_score(y_test, y_pred)),
        'confusion_matrix':        confusion_matrix(y_test, y_pred, labels=['REAL', 'AI_EDITED']).tolist(),
        'confusion_matrix_labels': ['REAL', 'AI_EDITED'],
        'classification_report':  classification_report(y_test, y_pred, output_dict=True),
        'n_train':                 len(X_train),
        'n_test':                  len(X_test),
        'n_failed_embeddings':    len(failed),
        'calibration':             calibration,
    }

    os.makedirs(_MODEL_DIR, exist_ok=True)
    joblib.dump(clf, _MODEL_PATH)
    with open(_META_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'Model saved to {_MODEL_PATH}')
    print(f'Held-out accuracy: {metrics["accuracy"]:.1%}  (n_test={metrics["n_test"]})')
    print(f'Confusion matrix {metrics["confusion_matrix_labels"]}:')
    for row in metrics['confusion_matrix']:
        print(f'  {row}')

    print(f'\nCalibration check (per-file confidence, correct vs incorrect predictions):')
    if calibration['correct_confidence_mean'] is not None:
        print(f'  Correct predictions   - mean confidence: {calibration["correct_confidence_mean"]:.1f}%  '
              f'(min: {calibration["correct_confidence_min"]:.1f}%)')
    if calibration['incorrect_confidence_mean'] is not None:
        print(f'  Incorrect predictions - mean confidence: {calibration["incorrect_confidence_mean"]:.1f}%  '
              f'(max: {calibration["incorrect_confidence_max"]:.1f}%)')
        if calibration['n_incorrect_high_confidence'] > 0:
            print(f'  \u26a0 FLAG: {calibration["n_incorrect_high_confidence"]} incorrect prediction(s) '
                  f'had confidence >= 80% - confidently WRONG, not just wrong.')

    return metrics
