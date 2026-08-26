"""
detector/audio_ai_gen_classifier.py

Domain-specific AI-generation/voice-clone classifier for AUDIO content.

WHY THIS EXISTS
----------------
audio_pipeline.py's current forensic_prob is built from five hand-crafted
spectral statistics (MFCC variance, spectral flatness, phase irregularity,
zero-crossing-rate abnormality, bandwidth anomaly). Real paired evaluation
this session (20 Aug 2026, 1,914 files: 870 real voices across 4 sources +
1,044 TTS clips across 5 sources) found only ONE of those five - bandwidth
anomaly - carries any real, reproducible separation; the rest are dead
(MFCC variance, phase irregularity) or actively backwards (spectral
flatness, confirmed inverted across all 5 fake-side batches). Reweighted
down to bandwidth-anomaly-only, the pipeline reaches ~67.5% overall
accuracy (80.0% specificity / 57.1% recall) - a real improvement over the
prior weighting, but nowhere near production-grade: an independent
neutral benchmark (Podonos, June 2026, 16 systems, private test set) shows
purpose-built commercial detectors (Resemble AI, Whispeak, Aurigin AI,
Pindrop) clearing 95-98% accuracy, while every system in that range is a
TRAINED MODEL - none of the hand-crafted-heuristic or untrained baseline
systems in that same benchmark cleared ~70%, which lines up almost
exactly with where this pipeline's heuristic approach tops out. That
benchmark's own pattern is the evidence base for this file's existence:
closing the gap needs a trained classifier on real embeddings, not
further tuning of five spectral statistics computed by hand.

This follows the exact same fix already applied twice on the image side
(vehicle_ai_gen_classifier.py, photo_edit_classifier.py): rather than
hand-tuning more heuristic features with a low ceiling, train a
lightweight classifier on your own labeled audio, using a frozen
pretrained speech model as a feature extractor (no GPU strictly required
for inference at this scale) and a small logistic-regression head on top.

MODEL CHOICE: facebook/wav2vec2-base - self-supervised, pretrained on
960 hours of unlabeled speech (LibriSpeech). Chosen over an ASR-fine-
tuned checkpoint (e.g. wav2vec2-base-960h) specifically because this
module wants a general acoustic representation, not one collapsed toward
transcription; a self-supervised base checkpoint is the more appropriate
feature extractor for a downstream classifier that isn't doing speech
recognition. ~95M parameters, comparable memory footprint to the CLIP
ViT-B/32 checkpoint already validated elsewhere in this codebase for the
2GB Render instance - NOT independently confirmed on Render itself as of
this file's creation (huggingface.co is unreachable from the dev sandbox
this was written in, same limitation router.py's own docstring already
flags) - the real test is the first live deploy, same caveat as router.py.

STATUS AS OF THIS FILE'S CREATION: UNTRAINED.
No model file exists yet. score_audio_domain() returns None until
train_and_save() (see train_audio_classifier.py for the CLI) has been run
against real labeled data and produced detector/models/audio_ai_gen_clf.joblib.

DATA NEEDED: two labeled sets, at least MIN_PER_CLASS each -
  - REAL:         genuine human voice recordings, no synthesis/cloning.
  - AI_GENERATED: TTS / voice-clone output.
The 1,914 files behind this session's audio validation (870 real across 4
sources, 1,044 fake across 5 sources) are a reasonable starting point if
the raw audio (not just the computed feature scores in the batch JSON
exports) is still available - those exports only ever carried the derived
stage_scores, not the source audio itself.

HONEST CEILING NOTE: even training on all 1,914 files, this remains a
much smaller and (per the source-batches' own uncertain TTS-engine
provenance) less deliberately diverse training set than what the 95%+
commercial systems in the Podonos benchmark were built on. Expect a real,
substantial improvement over the current ~67.5% heuristic baseline from
training this - do not expect 95-98% from a first pass; that tier
reflects large-scale, multi-engine training data and ongoing retraining
as new voice-cloning tools appear, not a one-time model swap. Validate
with real held-out numbers (see train_and_save()'s return value) before
deciding whether a bigger data-collection investment is worth it.

EVERY CALLER MUST TREAT None AS "SIGNAL UNAVAILABLE" AND DEGRADE
GRACEFULLY - same pattern as vehicle_domain_score / dl_ai_score being
None elsewhere in this codebase. None is NOT a score of 0; treating it
as 0 would silently assert "not AI-generated" with no basis.

DEPENDENCY NOTE: needs scikit-learn and joblib, same as
vehicle_ai_gen_classifier.py - confirm both are in requirements.txt
before relying on this in production. The sklearn/joblib imports inside
score_audio_domain() are wrapped in try/except ImportError specifically
so a missing dependency degrades to None rather than crashing every
AUDIO request - that guard is a safety net, not a substitute for
actually adding the dependency.
"""
import os
import json
import gc

import numpy as np
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

_WAV2VEC_MODEL_ID = 'facebook/wav2vec2-base'
_TARGET_SR = 16000  # wav2vec2's required sample rate - resampled here regardless of source file's native rate

# Same silence/duration guards already validated in audio_pipeline.py
# (16 Aug 2026 bug fixes) - reused rather than re-invented, since the
# failure modes (NaN/degenerate features on silent or too-short audio)
# apply just as much to an embedding model as to the hand-crafted
# features it's replacing here.
_MIN_RMS = 1e-4
_MIN_DURATION_SEC = 0.5

_MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')
_MODEL_PATH = os.path.join(_MODEL_DIR, 'audio_ai_gen_clf.joblib')
_META_PATH  = os.path.join(_MODEL_DIR, 'audio_ai_gen_clf_meta.json')

_CLASSES = ['AI_GENERATED', 'REAL']  # fixed, alphabetical - matches sklearn's default classes_ ordering


def _extract_wav2vec_embedding(filepath: str) -> np.ndarray:
    """
    Loads wav2vec2, extracts one audio file's embedding, fully releases
    the model before returning - same load-score-flush pattern as every
    other HF model in this codebase (see image_pipeline.py header comment
    and vehicle_ai_gen_classifier.py's identical docstring note: no two
    models resident at once on the 2GB Render instance).

    Embedding is the mean-pooled last_hidden_state across the time axis -
    a fixed-size (768-dim for wav2vec2-base) vector regardless of clip
    length, the standard approach for turning a variable-length sequence
    model's output into a single feature vector for a downstream
    classifier.

    Raises ValueError on silent/near-silent or too-short audio, same
    guards and same reasoning as audio_pipeline.py's analyze_audio() -
    an embedding computed from a signal with no meaningful energy or too
    few frames is not a meaningful embedding, and training or scoring on
    one would be worse than a clear failure.
    """
    audio, sr = librosa.load(filepath, sr=_TARGET_SR, mono=True)

    if sr <= 0 or len(audio) == 0:
        raise ValueError(f'{filepath}: no readable audio samples (corrupt or empty file)')

    duration = len(audio) / sr
    if duration < _MIN_DURATION_SEC:
        raise ValueError(
            f'{filepath}: audio too short to embed reliably ({duration:.2f}s, '
            f'need at least {_MIN_DURATION_SEC}s)'
        )

    audio_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if audio_rms < _MIN_RMS:
        raise ValueError(
            f'{filepath}: silent or near-silent (RMS={audio_rms:.2e}, need at '
            f'least {_MIN_RMS:.0e}) - an embedding from this is not meaningful'
        )

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(_WAV2VEC_MODEL_ID)
    model     = Wav2Vec2Model.from_pretrained(_WAV2VEC_MODEL_ID).to(device)
    model.eval()
    try:
        inputs = extractor(audio, sampling_rate=_TARGET_SR, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean-pool across the time dimension -> fixed-size embedding per clip
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    finally:
        del model, extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return embedding


def score_audio_domain(filepath: str):
    """
    Returns a 0-100 float (confidence the audio is AI-generated / a voice
    clone) if a trained model exists, otherwise None.

    Callers MUST handle None as "signal unavailable" - see module
    docstring. Only raises if a model file exists but is corrupt/
    unreadable, or if the audio itself is unusable (silent/too-short,
    same as analyze_audio()'s own guards) - both are real errors worth
    surfacing. Never raises for "not trained yet" or "dependency missing"
    - both are expected states right now and degrade to None silently by
    design.
    """
    if not os.path.exists(_MODEL_PATH):
        return None

    try:
        import joblib
    except ImportError:
        return None

    clf = joblib.load(_MODEL_PATH)
    embedding = _extract_wav2vec_embedding(filepath)
    proba = clf.predict_proba(embedding.reshape(1, -1))[0]
    ai_gen_index = list(clf.classes_).index('AI_GENERATED')
    return float(proba[ai_gen_index] * 100)


def train_and_save(real_audio_paths: list, ai_generated_audio_paths: list,
                    test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    OFFLINE TRAINING - not called during request handling. Run manually via
    train_audio_classifier.py once labeled audio exists for BOTH classes.

    Returns real held-out evaluation metrics (accuracy, confusion matrix,
    per-class precision/recall) - printed AND returned, never hidden. Per
    project standard: no fix ships on a reasoned guess, only on real
    evaluation results, and that applies to this classifier's own accuracy
    just as much as to any threshold change elsewhere in this codebase.
    Compare this run's held-out accuracy against the current heuristic
    baseline (~67.5% overall, 80.0% specificity / 57.1% recall, validated
    20 Aug 2026 on 1,914 files) before deciding whether to switch
    audio_pipeline.py over to this signal - a classifier trained on too
    little or too narrow data could plausibly score WORSE held-out than
    the existing heuristic despite the architecture upgrade, same
    generalization risk already confirmed once this session on the
    vehicle side (dl_ai's separation collapsing between generators) -
    don't assume "trained model" automatically beats "heuristic" without
    checking the actual number.

    Raises ValueError if either class has fewer than MIN_PER_CLASS audio
    files - training a classifier on fewer than that and deploying it
    silently would be worse than refusing to train at all.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import joblib

    MIN_PER_CLASS = 10
    if len(real_audio_paths) < MIN_PER_CLASS or len(ai_generated_audio_paths) < MIN_PER_CLASS:
        raise ValueError(
            f'Need at least {MIN_PER_CLASS} audio files per class to train a '
            f'meaningful classifier. Got {len(real_audio_paths)} REAL, '
            f'{len(ai_generated_audio_paths)} AI_GENERATED. Training refused.'
        )

    X, y, failed = [], [], []
    for path in real_audio_paths:
        try:
            X.append(_extract_wav2vec_embedding(path))
            y.append('REAL')
        except Exception as e:
            failed.append((path, str(e)))
    for path in ai_generated_audio_paths:
        try:
            X.append(_extract_wav2vec_embedding(path))
            y.append('AI_GENERATED')
        except Exception as e:
            failed.append((path, str(e)))

    if failed:
        print(f'WARNING: {len(failed)} audio files failed to embed and were skipped:')
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
    print(f'\nCompare against current heuristic baseline (67.5% overall, '
          f'80.0% specificity / 57.1% recall, 1,914-file validation) before '
          f'switching audio_pipeline.py over to this signal.')

    return metrics
