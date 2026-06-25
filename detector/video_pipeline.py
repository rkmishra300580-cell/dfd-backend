"""
Video analysis pipeline — continuous scoring replacing binary thresholds.
Catches: AI-generated video, face-swap video, edited frames, temporal manipulation.

Key fix: old pipeline produced only 0/25/50/75/100% outputs due to binary
threshold counting. Now uses continuous 0-100 scoring on every metric.
Added per-frame ELA to catch edited individual frames.
"""
import numpy as np
import cv2
import io
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.fftpack import fft2 as scipy_fft2, fftshift as scipy_fftshift

from .result import AnalysisResult
from .helpers import detect_faces, apply_graph_style


def _frame_ela(frame_rgb):
    """
    Run ELA on a single video frame.
    Returns ELA mean and regional inconsistency score (0-100).
    """
    try:
        pil_frame = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        pil_frame.save(buf, format='JPEG', quality=85)
        buf.seek(0)
        recomp = Image.open(buf).convert('RGB')
        ela_diff = ImageChops.difference(pil_frame.convert('RGB'), recomp)
        ela_arr  = np.array(ela_diff).astype(float)
        ela_gray = np.mean(ela_arr, axis=2)

        ela_mean = float(np.mean(ela_gray))
        h, w     = ela_gray.shape
        qh, qw   = h // 2, w // 2
        quads    = [
            float(np.mean(ela_gray[:qh, :qw])),
            float(np.mean(ela_gray[:qh, qw:])),
            float(np.mean(ela_gray[qh:, :qw])),
            float(np.mean(ela_gray[qh:, qw:])),
        ]
        regional = max(quads) - min(quads)
        # Score: high mean or high regional = suspicious
        score = float(np.clip(ela_mean * 2 + regional * 1.5, 0, 100))
        return ela_mean, score
    except Exception:
        return 0.0, 0.0


def analyze_video(filepath, R: AnalysisResult):
    R.pdf_text('VIDEO FORENSIC ANALYSIS REPORT', 'Title')
    apply_graph_style()

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError('Cannot open video file')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / fps if fps > 0 else 0

    R.add_stat('Resolution', f'{vid_w}x{vid_h}')
    R.add_stat('FPS',        f'{fps:.2f}')
    R.add_stat('Duration',   f'{duration:.1f}s')
    R.add_stat('Total Frames', total_frames)

    # Sample up to 20 frames evenly
    n_samples        = min(20, total_frames)
    frame_indices    = np.linspace(0, total_frames-1, n_samples, dtype=int)
    extracted_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            extracted_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not extracted_frames:
        R.add_indicator('[Video] No frames could be extracted')
        return 0.0

    indicators = []

    # ── Graph 1: Frame grid with face detection ──────────────────
    face_counts = []
    n_show      = min(6, len(extracted_frames))
    fig, axes   = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        if i < n_show:
            frame  = extracted_frames[i * (len(extracted_frames) // n_show)]
            faces  = detect_faces(frame, 0.4)
            face_counts.append(len(faces))
            visual = frame.copy()
            for (x,y,fw,fh) in faces:
                cv2.rectangle(visual,(x,y),(x+fw,y+fh),(88,166,255),3)
            ax.imshow(visual)
            ax.set_title(f'Frame {i} | {len(faces)} face(s)', color='#c9d1d9', fontsize=9)
        ax.axis('off')
    plt.suptitle('Sampled Frame Analysis', color='#58a6ff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    R.save_graph('video_frames.png', 'Sampled Frames',
                 'Sampled frames with face detection overlays.', important=True)
    plt.close(fig)

    # Also collect face counts for remaining frames
    for frame in extracted_frames[n_show:8]:
        faces = detect_faces(frame, 0.4)
        face_counts.append(len(faces))

    # ── FFT consistency across frames ───────────────────────────
    fft_scores = []
    for frame in extracted_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        fft_scores.append(float(np.std(np.log1p(np.abs(scipy_fftshift(scipy_fft2(gray)))))))

    fft_mean = float(np.mean(fft_scores))
    fft_std  = float(np.std(fft_scores))
    fft_cv   = fft_std / (fft_mean + 1e-6)  # coefficient of variation

    # Continuous score: lower CV = more suspicious (unnaturally consistent)
    # Real videos have CV > 0.05; deepfake videos often < 0.02
    fft_suspicion = float(np.clip((0.08 - fft_cv) / 0.08 * 100, 0, 100))

    R.add_stat('FFT Mean',        f'{fft_mean:.3f}')
    R.add_stat('FFT Std',         f'{fft_std:.3f}')
    R.add_stat('FFT CV',          f'{fft_cv:.4f}')
    R.add_stat('FFT Suspicion',   f'{fft_suspicion:.1f}%')

    if fft_suspicion > 60:
        indicators.append(f'Unnaturally consistent FFT across frames (CV={fft_cv:.3f}) — synthetic video suspected')
    elif fft_suspicion > 35:
        indicators.append(f'Moderately consistent frame FFT (CV={fft_cv:.3f})')

    # ── Temporal difference analysis ─────────────────────────────
    temporal_diffs = []
    for i in range(len(extracted_frames)-1):
        g1 = cv2.cvtColor(extracted_frames[i],   cv2.COLOR_RGB2GRAY).astype(float)
        g2 = cv2.cvtColor(extracted_frames[i+1], cv2.COLOR_RGB2GRAY).astype(float)
        temporal_diffs.append(float(np.mean(np.abs(g1 - g2))))

    if temporal_diffs:
        td_mean = float(np.mean(temporal_diffs))
        td_std  = float(np.std(temporal_diffs))
        td_cv   = td_std / (td_mean + 1e-6)

        # Real video: td_mean typically 5-30 depending on motion
        # Deepfake/still: td_mean very low or very uniform
        # Score on both low mean AND low variation
        low_motion_score    = float(np.clip((5.0 - td_mean) / 5.0 * 60, 0, 60))
        uniform_motion_score = float(np.clip((0.3 - td_cv) / 0.3 * 40, 0, 40))
        temporal_suspicion  = low_motion_score + uniform_motion_score

        R.add_stat('Temporal Diff Mean', f'{td_mean:.2f}')
        R.add_stat('Temporal Diff CV',   f'{td_cv:.3f}')
        R.add_stat('Temporal Suspicion', f'{temporal_suspicion:.1f}%')

        if temporal_suspicion > 60:
            indicators.append(f'Very low/uniform temporal motion (mean={td_mean:.1f}) — unnatural stillness or generated video')
        elif temporal_suspicion > 35:
            indicators.append(f'Moderately low temporal variation (mean={td_mean:.1f})')
    else:
        temporal_suspicion = 0.0
        td_mean = td_std = 0.0

    # ── Face count consistency ────────────────────────────────────
    face_suspicion = 0.0
    if face_counts:
        unique_counts   = len(set(face_counts))
        face_count_mean = float(np.mean(face_counts))
        # Perfect consistency with faces present = suspicious (real videos have occlusion)
        if unique_counts == 1 and face_count_mean > 0 and len(face_counts) >= 4:
            face_suspicion = 55.0
            indicators.append(f'Perfectly consistent face count across all frames ({int(face_count_mean)} faces) — unnatural')
        elif unique_counts <= 2 and face_count_mean > 0 and len(face_counts) >= 6:
            face_suspicion = 25.0

    R.add_stat('Face Count Variance', f'{len(set(face_counts))} unique values' if face_counts else 'N/A')
    R.add_stat('Face Suspicion',      f'{face_suspicion:.1f}%')

    # ── Per-frame ELA ─────────────────────────────────────────────
    # Detects frames that have been individually edited/replaced
    ela_frame_scores = []
    for frame in extracted_frames[:10]:  # limit to 10 frames for speed
        _, ela_s = _frame_ela(frame)
        ela_frame_scores.append(ela_s)

    ela_mean_score = float(np.mean(ela_frame_scores)) if ela_frame_scores else 0.0
    ela_max_score  = float(np.max(ela_frame_scores))  if ela_frame_scores else 0.0
    ela_suspicion  = float(np.clip(ela_mean_score * 0.6 + ela_max_score * 0.4, 0, 100))

    R.add_stat('Frame ELA Mean',   f'{ela_mean_score:.1f}')
    R.add_stat('Frame ELA Max',    f'{ela_max_score:.1f}')
    R.add_stat('ELA Suspicion',    f'{ela_suspicion:.1f}%')

    if ela_suspicion > 55:
        indicators.append(f'High ELA score on sampled frames (mean={ela_mean_score:.1f}) — individual frames may be edited')
    elif ela_suspicion > 30:
        indicators.append(f'Moderate ELA on sampled frames — possible frame-level editing')

    # ── Graph 2: Temporal analysis ───────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0,0].plot(fft_scores, marker='o', color='#58a6ff', linewidth=2)
    axes[0,0].fill_between(range(len(fft_scores)), fft_scores, alpha=0.15, color='#58a6ff')
    axes[0,0].axhline(y=fft_mean, color='#f85149', linestyle='--', label=f'Mean={fft_mean:.2f}')
    axes[0,0].set_title(f'FFT Std Across Frames (CV={fft_cv:.3f})', color='#c9d1d9')
    axes[0,0].set_xlabel('Frame Sample'); axes[0,0].legend(); axes[0,0].grid(True)

    if temporal_diffs:
        axes[0,1].plot(temporal_diffs, marker='o', color='#3fb950', linewidth=2)
        axes[0,1].fill_between(range(len(temporal_diffs)), temporal_diffs, alpha=0.15, color='#3fb950')
        axes[0,1].axhline(y=td_mean, color='#f85149', linestyle='--', label=f'Mean={td_mean:.2f}')
        axes[0,1].set_title(f'Temporal Difference (CV={td_cv:.3f})', color='#c9d1d9')
        axes[0,1].set_xlabel('Frame Pair'); axes[0,1].legend(); axes[0,1].grid(True)

    if ela_frame_scores:
        axes[1,0].bar(range(len(ela_frame_scores)), ela_frame_scores,
                      color=['#f85149' if s > 50 else '#d29922' if s > 25 else '#3fb950' for s in ela_frame_scores])
        axes[1,0].axhline(y=50, color='#f85149', linestyle='--', label='Suspicion threshold')
        axes[1,0].set_title(f'Per-Frame ELA Score (mean={ela_mean_score:.1f})', color='#c9d1d9')
        axes[1,0].set_xlabel('Frame'); axes[1,0].set_ylabel('ELA Score'); axes[1,0].legend()

    # Score dashboard
    score_labels  = ['FFT\nConsistency', 'Temporal\nMotion', 'Face\nCount', 'Frame\nELA']
    score_values  = [fft_suspicion, temporal_suspicion, face_suspicion, ela_suspicion]
    bar_colors    = ['#f85149' if v >= 50 else '#d29922' if v >= 25 else '#3fb950' for v in score_values]
    bars = axes[1,1].bar(score_labels, score_values, color=bar_colors, width=0.5)
    axes[1,1].axhline(y=50, color='#f85149', linestyle='--', linewidth=1.5, label='Suspicion threshold')
    for bar, val in zip(bars, score_values):
        axes[1,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                       f'{val:.0f}', ha='center', color='#c9d1d9', fontweight='bold')
    axes[1,1].set_ylim(0, 115)
    axes[1,1].set_title('Video Forensic Dashboard', color='#c9d1d9')
    axes[1,1].legend()

    plt.suptitle('Video Forensic Analysis', color='#58a6ff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    R.save_graph('video_temporal.png', 'Temporal & ELA Analysis',
                 'FFT consistency, temporal motion, face count, and per-frame ELA scores.', important=True)
    plt.close(fig)

    for ind in indicators:
        R.add_indicator(f'[Video] {ind}')

    # ── Continuous fusion score ───────────────────────────────────
    # Weighted combination — all four metrics contribute
    final_score = float(np.clip(
        fft_suspicion      * 0.30 +
        temporal_suspicion * 0.30 +
        face_suspicion     * 0.15 +
        ela_suspicion      * 0.25,
        0, 100
    ))

    R.add_stat('FFT Score',       f'{fft_suspicion:.1f}%')
    R.add_stat('Temporal Score',  f'{temporal_suspicion:.1f}%')
    R.add_stat('Face Score',      f'{face_suspicion:.1f}%')
    R.add_stat('ELA Score',       f'{ela_suspicion:.1f}%')
    R.add_stat('Final Video Score', f'{final_score:.1f}%')
    R.payload['stage_scores']['video_temporal'] = round(final_score, 1)

    return final_score
