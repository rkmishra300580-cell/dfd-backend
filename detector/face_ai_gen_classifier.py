"""
detector/face_ai_gen_classifier.py

Domain-specific AI-generation classifier for FACE-bucket images.

SCOPE - READ BEFORE COLLECTING DATA
------------------------------------
This targets the AI_GENERATED track only: real photo of a person vs.
wholesale AI-generated face (StyleGAN/diffusion/Midjourney-style, no real
photo underneath). It is NOT for the DEEPFAKE/face-swap track - that's a
different classification problem (real photo of a real person, with the
face swapped/replaced) and already has a dedicated specialist model
(prithivMLmods, `deep_learning` in stage_scores) rather than a generic
mismatched tool, so it doesn't have the same domain-transfer problem this
file exists to fix. Training data for THIS file must be:
  - REAL:         genuine, unedited photos of real people.
  - AI_GENERATED: wholesale AI-generated faces (no real photo underneath).
Do NOT use deepfake/face-swap images for either class here - that's a
different task with a different model already assigned to it.

WHY THIS EXISTS
----------------
The FACE-bucket AI_GENERATED track currently relies on the same two
general-purpose detectors used elsewhere in this pipeline (Organika/
sdxl-detector, umm-maybe/AI-image-detector - see image_pipeline.py's
dl_detector()). These were confirmed, on VEHICLE-bucket content this
session (26 Aug 2026), to show real generator-specific instability: the
Organika/sdxl-detector member showed +33.0 mean separation between real
and AI-generated content when the fakes were (likely) SDXL-family output,
collapsing to +2.7 - essentially no separation - when the fakes were
Midjourney-generated instead. Organika/sdxl-detector's own model card
already states it was fine-tuned specifically on SDXL-regeneration pairs
and may not generalize to other generators.

This is the same underlying ensemble used for FACE-bucket AI_GENERATED
scoring, so the same generator-instability risk applies there by
extension - NOT yet independently confirmed on face content specifically
(no FACE-bucket batch test has been run this session), only inferred from
the identical underlying models and the same architectural pattern. This
file follows the same fix already applied to VEHICLE-bucket content
(vehicle_ai_gen_classifier.py, validated 26-27 Aug 2026 across 5
independent batches, 100% held-out accuracy each time) and AUDIO
(audio_ai_gen_classifier.py, 97-98% held-out accuracy): train a
lightweight classifier on your own labeled domain data, using a frozen
CLIP model as a feature extractor, rather than continuing to rely on
generic pretrained detectors with an unconfirmed generalization ceiling
for this specific content type.

STATUS AS OF THIS FILE'S CREATION: UNTRAINED, NOT YET INTEGRATED.
No model file exists yet. score_face_domain() returns None until
train_and_save() (see train_face_classifier.py for the CLI, or the
matching Colab notebook) has been run against real labeled data and
produced detector/models/face_ai_gen_clf.joblib. Unlike
vehicle_ai_gen_classifier.py, this is also NOT YET WIRED into
image_pipeline.py/helpers.py's composite formula - that integration is a
deliberate separate step, done only once real held-out accuracy is
confirmed (same two-step sequence followed for the audio classifier).

EVERY CALLER MUST TREAT None AS "SIGNAL UNAVAILABLE" AND DEGRADE
GRACEFULLY - same pattern as vehicle_domain_score / dl_ai_score being
None elsewhere in this codebase. None is NOT a score of 0; treating it
as 0 would silently assert "not AI-generated" with no basis.

DEPENDENCY NOTE: needs scikit-learn and joblib, same as
vehicle_ai_gen_classifier.py and audio_ai_gen_classifier.py - confirm
both are in requirements.txt before relying on this in production. The
sklearn/joblib imports inside score_face_domain() are wrapped in
try/except ImportError specifically so a missing dependency degrades to
None rather than crashing every FACE-bucket request - that guard is a
safety net, not a substitute for actually adding the dependency.
"""
import os
import json
import gc

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

# Same CLIP checkpoint already used by router.py's content-type router,
# vehicle_ai_gen_classifier.py, and photo_edit_classifier.py - reusing a
# choice already validated for memory footprint on the 2GB Render
# instance, not introducing a new model to load/flush.
_CLIP_MODEL_ID = 'openai/clip-vit-base-patch32'

_MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')
_MODEL_PATH = os.path.join(_MODEL_DIR, 'face_ai_gen_clf.joblib')
_META_PATH  = os.path.join(_MODEL_DIR, 'face_ai_gen_clf_meta.json')

_CLASSES = ['AI_GENERATED', 'REAL']  # fixed, alphabetical - matches sklearn's default classes_ ordering


def _extract_clip_embedding(pil_image) -> np.ndarray:
    """
    Loads CLIP, extracts one image's embedding, fully releases the model
    before returning - same load-score-flush pattern as every other HF
    model in this codebase (see image_pipeline.py header comment: no two
    models resident at once on the 2GB Render instance).
    """
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
    model     = CLIPModel.from_pretrained(_CLIP_MODEL_ID).to(device)
    model.eval()
    try:
        inputs = processor(images=pil_image.convert('RGB'), return_tensors='pt').to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        # BUG FIX (confirmed on vehicle_ai_gen_classifier.py, 26 Aug 2026):
        # get_image_features() is documented to return a raw tensor, but
        # was confirmed to return a wrapped model-output object instead
        # under certain transformers versions in Colab ("'BaseModelOutput
        # WithPooling' object has no attribute 'cpu'"). Applied here
        # proactively from the start rather than waiting to hit the same
        # crash - handles both possible return shapes.
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


def score_face_domain(pil_image):
    """
    Returns a 0-100 float (confidence the image is a wholesale AI-generated
    face, NOT a face-swap/deepfake) if a trained model exists, otherwise
    None.

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
    ai_gen_index = list(clf.classes_).index('AI_GENERATED')
    return float(proba[ai_gen_index] * 100)


def train_and_save(real_image_paths: list, ai_generated_image_paths: list,
                    test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    OFFLINE TRAINING - not called during request handling. Run manually via
    train_face_classifier.py (or the matching Colab notebook) once labeled
    face images exist for BOTH classes.

    Returns real held-out evaluation metrics (accuracy, confusion matrix,
    per-class precision/recall) - printed AND returned, never hidden. Per
    project standard: no fix ships on a reasoned guess, only on real
    evaluation results.

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
    if len(real_image_paths) < MIN_PER_CLASS or len(ai_generated_image_paths) < MIN_PER_CLASS:
        raise ValueError(
            f'Need at least {MIN_PER_CLASS} images per class to train a '
            f'meaningful classifier. Got {len(real_image_paths)} REAL, '
            f'{len(ai_generated_image_paths)} AI_GENERATED. Training refused.'
        )

    X, y, failed = [], [], []
    for path in real_image_paths:
        try:
            X.append(_extract_clip_embedding(PILImage.open(path)))
            y.append('REAL')
        except Exception as e:
            failed.append((path, str(e)))
    for path in ai_generated_image_paths:
        try:
            X.append(_extract_clip_embedding(PILImage.open(path)))
            y.append('AI_GENERATED')
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

    # CALIBRATION FIX (28 Aug 2026) - a bare LogisticRegression's
    # predict_proba() is NOT guaranteed to be well-calibrated (its output
    # is a sigmoid over a linear decision function, not a property that
    # cross-entropy training directly optimizes for probability accuracy).
    # Confirmed a real, reproducible problem here, not a theoretical
    # concern: 3 consecutive batches from a second dataset (27-28 Aug
    # 2026, ~630-660 files each) each produced at least one confidently-
    # wrong prediction (>=80% confidence, max observed 99.93% - i.e.
    # essentially certain and completely wrong), with the confidently-
    # wrong RATE increasing across the first two of those three batches
    # before this fix. This directly undermines the per-file 'confidence'
    # number shown to end users (helpers.py's classify_dominant(),
    # distance-from-threshold on this classifier's own predict_proba()
    # output once wired into production) - a near-100%-confidence wrong
    # result is the single worst-case outcome for that number's
    # trustworthiness.
    #
    # CalibratedClassifierCV wraps the base classifier and recalibrates
    # its probability output via cross-validation - it does NOT change
    # what the base classifier learns or its accuracy on held-out data,
    # only how honestly its predict_proba() reflects real confidence.
    # method='sigmoid' (Platt scaling) chosen over 'isotonic' - isotonic
    # is more flexible but needs more data to avoid overfitting the
    # calibration itself (sklearn's own guidance: prefer sigmoid below
    # roughly 1,000 samples); every batch trained here so far has been in
    # the 500-700 range, well under that.
    #
    # cv fold count is capped by the smallest class's training-set size,
    # not hardcoded to 5 - MIN_PER_CLASS only guarantees 10 images per
    # class BEFORE the train/test split and BEFORE any embedding
    # failures, so a small or unlucky batch could leave fewer than 5
    # per class in y_train; a hardcoded cv=5 would then raise rather
    # than degrade gracefully.
    #
    # score_face_domain() (the function that actually runs in
    # production) needs NO changes for this - CalibratedClassifierCV
    # exposes the identical .predict_proba()/.classes_ interface a bare
    # LogisticRegression does, so every downstream caller is unaffected.
    base_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    min_class_count = min(np.bincount(np.unique(y_train, return_inverse=True)[1]))
    cv_folds = max(2, min(5, min_class_count))
    clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=cv_folds)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # CALIBRATION CHECK (27 Aug 2026) - added per product decision that the
    # per-file 'confidence' number shown to end users (helpers.py's
    # classify_dominant(), distance-from-threshold on this classifier's own
    # predict_proba() output once wired into production) needs to actually
    # mean something. Accuracy alone can't tell you that: a model can be
    # 90% accurate while still being CONFIDENTLY WRONG on its misses (e.g.
    # 95% sure on a file it got wrong) - which would feed a misleadingly
    # high confidence number to an end user on exactly the cases where they
    # should be told to doubt the result. A model that's UNCERTAIN on its
    # misses (confidence near 50%) is behaving honestly - the wrongness and
    # the low confidence agree with each other.
    #
    # This reports, for the held-out test set: mean/min/max predict_proba()
    # confidence (distance from 50%, same shape as the deployed confidence
    # number) split by whether the prediction was correct or not. Healthy
    # pattern: correct predictions cluster near the extremes (high
    # confidence), incorrect predictions cluster nearer 50% (low
    # confidence). Red flag: incorrect predictions with high confidence
    # scores - means predict_proba() isn't well-calibrated for this model/
    # dataset, and the raw probability shouldn't be trusted as a
    # per-file confidence signal without further calibration work
    # (e.g. sklearn's CalibratedClassifierCV) before relying on it.
    proba_test = clf.predict_proba(X_test)
    ai_gen_col = list(clf.classes_).index('AI_GENERATED')
    proba_ai_gen = proba_test[:, ai_gen_col]
    # "confidence" here mirrors the deployed shape: distance from 50/50,
    # rescaled to 0-100 (0 = coin flip, 100 = certain either direction) -
    # not the raw class probability, which conflates "confident REAL" and
    # "confident AI_GENERATED" into opposite ends of a 0-100 scale instead
    # of both reading as "high confidence".
    per_file_confidence = np.abs(proba_ai_gen - 0.5) * 200
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
        'confusion_matrix':        confusion_matrix(y_test, y_pred, labels=['REAL', 'AI_GENERATED']).tolist(),
        'confusion_matrix_labels': ['REAL', 'AI_GENERATED'],
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
                  f'had confidence >= 80% - the model was confidently WRONG on these, not just wrong. '
                  f'This is worse than a low-confidence miss and worth a closer look before trusting '
                  f'this model\'s confidence numbers.')
        elif calibration['correct_confidence_mean'] is not None and \
             calibration['incorrect_confidence_mean'] < calibration['correct_confidence_mean']:
            print(f'  Healthy pattern: incorrect predictions show lower confidence than correct ones - '
                  f'the model is appropriately unsure when it\'s wrong, not confidently wrong.')

    return metrics
