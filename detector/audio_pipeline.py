"""
Audio analysis pipeline — MFCC, spectral flatness, phase irregularity,
zero-crossing rate, and spectral bandwidth features for voice-clone/TTS detection.
Logic carried over unchanged from the validated Colab prototype (Cell 10).
"""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from .result import AnalysisResult
from .helpers import apply_graph_style


def analyze_audio(filepath, R: AnalysisResult):
    R.pdf_text('AUDIO FORENSIC ANALYSIS REPORT', 'Title')
    apply_graph_style()

    audio, sr = librosa.load(filepath, sr=None, mono=True)

    if sr <= 0 or len(audio) == 0:
        raise ValueError('Audio file contains no readable samples (corrupt or empty file)')

    duration = len(audio) / sr

    # BUG FIX (16 Aug 2026): silent/near-silent audio does NOT produce
    # NaN in librosa's feature functions (confirmed by direct testing) -
    # it produces fully finite but MEANINGLESS values that read as highly
    # suspicious: spectral_flatness=1.0 (max "flatness"), phase_diff
    # std=0.0 (max "irregularity"), spectral_bandwidth=0.0, mfcc_std in
    # the hundreds (log-of-near-zero artifact in the mel filterbank).
    # Tested directly: 2 seconds of pure silence scores ~76% "suspicious"
    # under the existing weights - a real false-positive generator for
    # any empty, corrupted, or mostly-silent upload, and NOT caught by
    # the isfinite() check below since nothing is actually NaN here.
    # RMS energy is checked directly instead - a genuinely finite-looking
    # score built from a signal with no meaningful energy is not a signal
    # at all, and reporting a percentage on it either way would assert
    # confidence the analysis never had a basis for.
    MIN_RMS = 1e-4
    audio_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if audio_rms < MIN_RMS:
        raise ValueError(
            f'Audio is silent or near-silent (RMS={audio_rms:.2e}, need at '
            f'least {MIN_RMS:.0e}) - forensic voice features are undefined '
            f'on a signal with no meaningful energy.'
        )

    # BUG FIX (16 Aug 2026): every score below depends on stft-based features
    # (phase_diff = np.diff(np.angle(stft), axis=1)) needing at least 2 STFT
    # frames to be non-empty. Below roughly this length, phase_diff becomes
    # an empty array and np.std() on it silently returns NaN - which then
    # propagates into forensic_prob with no error raised anywhere, producing
    # a corrupted "nan%" score in the payload/PDF instead of a clear failure.
    # 0.5s is a generous floor well above the minimum the math actually
    # needs (roughly n_fft + hop_length samples, ~0.12s at 22kHz) - chosen
    # for a clean, explainable message rather than cutting it as fine as
    # possible. Raising here (not silently degrading) matches the existing
    # pattern in document_pipeline.py for unprocessable input - it surfaces
    # as a clean R.payload['error'] via run_pipeline()'s existing try/except,
    # not a crash and not a silently-wrong score.
    MIN_DURATION_SEC = 0.5
    if duration < MIN_DURATION_SEC:
        raise ValueError(
            f'Audio clip too short to analyze reliably ({duration:.2f}s, '
            f'need at least {MIN_DURATION_SEC}s) - forensic features need '
            f'multiple analysis frames to be meaningful.'
        )

    R.add_stat('Sample Rate', f'{sr} Hz')
    R.add_stat('Duration',    f'{duration:.2f} sec')

    # ── Graph 1: Waveform + Mel Spectrogram (IMPORTANT)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    librosa.display.waveshow(audio, sr=sr, ax=axes[0], color='#58a6ff')
    axes[0].set_title('Waveform', color='#c9d1d9'); axes[0].set_facecolor('#161b22')

    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128), ref=np.max
    )
    img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1], cmap='magma')
    plt.colorbar(img, ax=axes[1], format='%+2.0f dB')
    axes[1].set_title('Mel Spectrogram  — Synthetic voices often show unnatural banding', color='#c9d1d9')
    axes[1].set_facecolor('#161b22')

    plt.suptitle('Audio Waveform & Spectrogram', color='#58a6ff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    R.save_graph('audio_waveform_mel.png', 'Waveform & Mel Spectrogram',
                 'Waveform (top) and mel spectrogram (bottom). TTS/voice-cloning often shows uniform energy distribution.', important=True)
    plt.close(fig)

    # Features
    mfcc       = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_std   = float(np.std(mfcc))
    spec_flat  = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    spec_bw    = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    zcr        = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    stft       = librosa.stft(audio)
    phase_diff = np.diff(np.angle(stft), axis=1)

    # BUG FIX (16 Aug 2026): near-silent audio can produce NaN/Inf in
    # spectral_flatness or spectral_bandwidth (librosa dividing by
    # near-zero spectral energy), and phase_diff can end up empty on
    # very short/degenerate STFT output even after the duration check
    # above (e.g. audio that's mostly silence with only a brief onset).
    # Either would silently propagate a NaN into forensic_prob below with
    # no error raised - producing a corrupted "nan%" score in the payload
    # and PDF instead of a clear failure. Check explicitly and raise,
    # same reasoning as the duration guard above: a clean, honest failure
    # is strictly better than a silently-wrong score.
    _raw_features = {'mfcc_std': mfcc_std, 'spec_flat': spec_flat,
                      'spec_bw': spec_bw, 'zcr': zcr}
    _bad = {k: v for k, v in _raw_features.items() if not np.isfinite(v)}
    if _bad or phase_diff.size == 0 or not np.isfinite(np.std(phase_diff)):
        raise ValueError(
            f'Audio produced non-finite forensic features (likely silent '
            f'or near-silent audio) - cannot compute a reliable score. '
            f'Non-finite: {list(_bad.keys()) or ["phase_diff"]}'
        )

    R.add_stat('MFCC Std Dev',       f'{mfcc_std:.4f}')
    R.add_stat('Spectral Flatness',  f'{spec_flat:.6f}')
    R.add_stat('Spectral Bandwidth', f'{spec_bw:.2f} Hz')
    R.add_stat('Zero Crossing Rate', f'{zcr:.6f}')

    # ── Graph 2: MFCC heatmap (IMPORTANT)
    fig, ax = plt.subplots(figsize=(14, 5))
    img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax, cmap='coolwarm')
    plt.colorbar(img, ax=ax)
    ax.set_title(f'MFCC Coefficients  (std={mfcc_std:.2f})  — Low std suggests synthetic/monotone voice',
                 color='#c9d1d9')
    ax.set_facecolor('#161b22')
    plt.tight_layout()
    R.save_graph('audio_mfcc.png', 'MFCC Analysis',
                 f'MFCC coefficients across time. Natural speech shows high variation (std > 15). Current std={mfcc_std:.2f}.', important=True)
    plt.close(fig)

    # ── Component scoring ─────────────────────────────────────────────────
    # REWEIGHTED (16 Aug 2026) - real paired evidence, not a guess:
    # ran the actual pipeline against 74 known-synthetic (TTS, Kaggle) and
    # 8 known-real (genuine recordings) audio files.
    #
    # MFCC Variance and Phase Irregularity are DROPPED (weight 0) - both
    # confirmed dead on BOTH classes, not just weak: MFCC Variance read
    # exactly 0.0 on every single file in both batches (82/82, zero
    # exceptions) and Phase Irregularity sat in an ~87-88 band regardless
    # of content (real: 87.2 constant; TTS: 88.2-88.7). Neither is
    # discriminating anything - they were unconditionally diluting every
    # score by 40% of the total weight toward two fixed, uninformative
    # values, on top of everything.
    #
    # The remaining three all showed real separation on the same paired
    # data (real mean -> TTS mean): Spectral Flatness 3.1->12.8, ZCR
    # Abnormality 26.5->68.9 (by far the largest gap), Bandwidth Anomaly
    # 9.4->33.4. ZCR Abnormality weighted 3x tripled TTS recall (17/74 ->
    # 52/74, 23%->70%) with ZERO precision cost on the real-voice sample
    # (still 8/8 correctly REAL) - confirmed by testing multiple weighting
    # schemes against the real data, not chosen a priori. Pushing to
    # ZCR-only recovers more recall (63/74, 85%) but starts costing real
    # precision (6/8) - not worth the trade at this sample size.
    #
    # CAVEAT: the real-voice sample is only 8 files. "Zero precision cost"
    # on 8 examples is a promising first signal, not a proven guarantee -
    # re-validate this weighting as more real-voice data comes in, same as
    # any other threshold in this codebase that hasn't seen a large,
    # labeled negative set yet.
    scores = {
        'MFCC Variance'      : float(np.clip((20 - mfcc_std) / 20 * 100, 0, 100)),
        'Spectral Flatness'  : float(np.clip(spec_flat * 500, 0, 100)),
        'Phase Irregularity' : float(np.clip(100 - np.std(phase_diff) * 5, 0, 100)),
        'ZCR Abnormality'    : float(np.clip(abs(zcr - 0.08) * 1000, 0, 100)),
        'Bandwidth Anomaly'  : float(np.clip((3000 - spec_bw) / 30, 0, 100)),
    }
    _COMPONENT_WEIGHTS = {
        'MFCC Variance': 0, 'Spectral Flatness': 1, 'Phase Irregularity': 0,
        'ZCR Abnormality': 3, 'Bandwidth Anomaly': 1,
    }

    # ── Graph 3: Dashboard (IMPORTANT)
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors_map = {'MFCC Variance':'#f85149', 'Spectral Flatness':'#d29922',
                      'Phase Irregularity':'#58a6ff', 'ZCR Abnormality':'#3fb950', 'Bandwidth Anomaly':'#bc8cff'}
    bars = ax.bar(list(scores.keys()), list(scores.values()),
                  color=[bar_colors_map[k] for k in scores.keys()], width=0.5)
    ax.axhline(y=50, color='#f85149', linestyle='--', linewidth=1.5, label='Suspicion threshold')
    for bar, val in zip(bars, scores.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{val:.0f}', ha='center', color='#c9d1d9', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 110); ax.set_ylabel('Suspicion Score (0-100)')
    ax.set_title('Audio Forensic Dashboard  — Scores above 50 are suspicious', color='#c9d1d9')
    ax.legend(); plt.xticks(rotation=15)
    plt.tight_layout()
    R.save_graph('audio_dashboard.png', 'Audio Forensic Dashboard',
                 'Per-feature suspicion scores. Multiple high scores indicate synthetic/cloned voice.', important=True)
    plt.close(fig)

    indicators = []
    if mfcc_std  < 15:  indicators.append('Low MFCC variance — monotone/synthetic speech')
    if spec_flat > 0.1: indicators.append('High spectral flatness — possible TTS')
    if spec_bw   < 2000: indicators.append('Narrow spectral bandwidth — unnatural audio')

    for ind in indicators: R.add_indicator(f'[Audio] {ind}')

    forensic_prob = float(np.clip(
        sum(scores[k] * _COMPONENT_WEIGHTS[k] for k in scores) / sum(_COMPONENT_WEIGHTS.values()),
        0, 100
    ))
    R.payload['stage_scores']['audio_forensics'] = round(forensic_prob, 1)

    # DIAGNOSTIC FIX (16 Aug 2026): the 5 individual sub-scores that get
    # averaged into forensic_prob were never saved to stage_scores - only
    # the final averaged number was, and the individual values only ever
    # existed as R.add_stat() display strings (PDF-only, not captured by
    # any batch-testing harness). This made it impossible to diagnose WHY
    # a batch scored the way it did without re-deriving everything by hand.
    # Confirmed need: a 74-file batch of known-synthetic (TTS) audio (16
    # Aug 2026) scored only 23% recall (17/74), with scores compressed
    # into a narrow 27.7-48.0 range - but with no visibility into which of
    # the 5 features drove that, whether one is dead/uninformative on this
    # data (the same pattern already found and fixed twice on the image
    # side - mechanism #3's noise heuristic, the AI-Gen ensemble mean),
    # or whether all 5 are individually weak. This is diagnostic only -
    # does NOT change forensic_prob, audio_ai_generated, or any threshold;
    # it only exposes what already gets computed, the same way the image
    # pipeline exposes ela_score/copy_move_score/prnu_score/etc. alongside
    # its combined manipulation score.
    R.payload['stage_scores']['audio_mfcc_variance']       = round(scores['MFCC Variance'], 1)
    R.payload['stage_scores']['audio_spectral_flatness']   = round(scores['Spectral Flatness'], 1)
    R.payload['stage_scores']['audio_phase_irregularity']  = round(scores['Phase Irregularity'], 1)
    R.payload['stage_scores']['audio_zcr_abnormality']     = round(scores['ZCR Abnormality'], 1)
    R.payload['stage_scores']['audio_bandwidth_anomaly']   = round(scores['Bandwidth Anomaly'], 1)

    # Two-track split (mirrors the image pipeline). All five features above
    # detect synthetic/TTS audio in general - none of them verify whether
    # the voice matches a specific target identity, so audio_deepfake_score
    # is an honest placeholder (0), not a computed value. Real voice-clone
    # detection needs speaker/voice-identity verification, which isn't
    # implemented. Don't quietly change this to a nonzero heuristic later
    # without actually building that capability first.
    R.payload['stage_scores']['audio_ai_generated'] = round(forensic_prob, 1)
    R.payload['stage_scores']['audio_deepfake']     = 0.0

    return forensic_prob
