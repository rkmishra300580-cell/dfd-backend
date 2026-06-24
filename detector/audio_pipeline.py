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
    duration  = len(audio) / sr

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

    # Score components
    scores = {
        'MFCC Variance'      : float(np.clip((20 - mfcc_std) / 20 * 100, 0, 100)),
        'Spectral Flatness'  : float(np.clip(spec_flat * 500, 0, 100)),
        'Phase Irregularity' : float(np.clip(100 - np.std(phase_diff) * 5, 0, 100)),
        'ZCR Abnormality'    : float(np.clip(abs(zcr - 0.08) * 1000, 0, 100)),
        'Bandwidth Anomaly'  : float(np.clip((3000 - spec_bw) / 30, 0, 100)),
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

    forensic_prob = float(np.clip(np.mean(list(scores.values())), 0, 100))
    R.payload['stage_scores']['audio_forensics'] = round(forensic_prob, 1)

    return forensic_prob
