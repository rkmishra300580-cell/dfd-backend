"""
detector/vehicle_ai_gen_classifier.py

Domain-specific AI-generation classifier for VEHICLE-bucket images.

WHY THIS EXISTS
----------------
The two general-purpose AI-generation detectors used elsewhere in this
pipeline (Organika/sdxl-detector, umm-maybe/AI-image-detector - see
image_pipeline.py's dl_detector()) are trained on curated art/photography
datasets, never on vehicle-damage photos. Real evaluation data (9 labeled
vehicle-claim photos, 4 Aug 2026) showed this ensemble running
systematically hot on real, WhatsApp-compressed vehicle-damage photos:
dl_ai_generated scored 43-85% on photos that were misclassified
AI_GENERATED, vs 26-35% on photos correctly classified REAL - a domain-
transfer failure, not noise (see helpers.py's DL_AI_FLOOR /
DL_AI_CONCLUSIVE_THRESHOLD comments for the fix that scoped those two
generic-ensemble mechanisms to FACE content only, 4 Aug 2026).

Published research on AI-image-detection generalization (NTIRE 2026 /
cross-dataset benchmark literature) consistently finds that training-data
domain alignment explains more of a detector's real-world accuracy than
which architecture it uses. This module follows that finding directly:
rather than searching for yet another generic pretrained model and hoping
it transfers, it trains a lightweight classifier on your own vehicle-
domain images, using a frozen CLIP model as a feature extractor (cheap:
no GPU needed, trains in minutes once data exists) and a small logistic-
regression head on top.

STATUS AS OF THIS FILE'S CREATION: UNTRAINED.
No model file exists yet, because there is no labeled AI-generated-vehicle
training data yet (only 7 real-vehicle examples exist in the project's
current labeled set; zero confirmed AI-generated-vehicle examples).
score_vehicle_domain() returns None until train_and_save() (see
train_vehicle_classifier.py for the CLI) has been run against real
labeled data and produced detector/models/vehicle_ai_gen_clf.joblib.

EVERY CALLER MUST TREAT None AS "SIGNAL UNAVAILABLE" AND DEGRADE
GRACEFULLY - the same pattern already used elsewhere in this codebase for
dl_ai_score is None. None is NOT a score of 0; treating it as 0 would
silently assert "not AI-generated" with no basis.

DEPENDENCY NOTE: this module needs scikit-learn and joblib, which were
NOT in requirements.txt as of this file's creation. Add both before
relying on this in production. The sklearn/joblib imports inside
score_vehicle_domain() are wrapped in try/except ImportError specifically
so a missing dependency degrades to None (same as "not trained yet")
rather than crashing every VEHICLE-bucket request - but that guard is a
safety net, not a substitute for actually adding the dependency.
"""
import os
import json
import gc

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

# Same CLIP checkpoint already used by router.py's content-type router -
# reusing a choice already validated for memory footprint on this 2GB
# Render instance, not introducing a new model to load/flush.
_CLIP_MODEL_ID = 'openai/clip-vit-base-patch32'

_MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')
_MODEL_PATH = os.path.join(_MODEL_DIR, 'vehicle_ai_gen_clf.joblib')
_META_PATH  = os.path.join(_MODEL_DIR, 'vehicle_ai_gen_clf_meta.json')

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
        # BUG FIX (26 Aug 2026): get_image_features() is documented to return
        # a raw tensor, but was confirmed (Colab training run, real error:
        # "'BaseModelOutputWithPooling' object has no attribute 'cpu'") to
        # return a wrapped model-output object instead under the transformers
        # version installed there - a version-compatibility difference, not
        # a data problem (the error was identical across every failed file,
        # not varying per-image the way a corrupt-file error would).
        # Handles both shapes rather than assuming one, so this doesn't
        # silently break again on a different transformers version either way.
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


def score_vehicle_domain(pil_image):
    """
    Returns a 0-100 float (confidence the image is AI-generated, vehicle
    domain) if a trained model exists, otherwise None.

    Callers MUST handle None as "signal unavailable" - see module
    docstring. Only raises if a model file exists but is corrupt/
    unreadable (a real error worth surfacing), never for "not trained yet"
    or "dependency missing" - both of those are expected states right now
    and degrade to None silently by design.
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
    train_vehicle_classifier.py once labeled vehicle-domain images exist
    for BOTH classes.

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

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    metrics = {
        'accuracy':                float(accuracy_score(y_test, y_pred)),
        'confusion_matrix':        confusion_matrix(y_test, y_pred, labels=['REAL', 'AI_GENERATED']).tolist(),
        'confusion_matrix_labels': ['REAL', 'AI_GENERATED'],
        'classification_report':  classification_report(y_test, y_pred, output_dict=True),
        'n_train':                 len(X_train),
        'n_test':                  len(X_test),
        'n_failed_embeddings':    len(failed),
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

    return metrics
