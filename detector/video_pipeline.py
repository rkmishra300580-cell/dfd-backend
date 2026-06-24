"""
Video analysis pipeline — frame sampling, per-frame FFT consistency,
temporal-difference analysis, and face-count consistency.
Logic carried over unchanged from the validated Colab prototype (Cell 9).
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import fft2 as scipy_fft2, fftshift as scipy_fftshift

from .result import AnalysisResult
from .helpers import detect_faces, apply_graph_style


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

    R.add_stat('Resolution', f'{vid_w}x{vid_h}')
    R.add_stat('FPS', f'{fps:.2f}')
    R.add_stat('Total Frames', total_frames)

    frame_indices    = np.linspace(0, total_frames-1, min(20, total_frames), dtype=int)
    extracted_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok: extracted_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    # Face detection across frames
    face_counts = []
    for frame in extracted_frames[:8]:
        faces = detect_faces(frame, min_confidence=0.4)
        face_counts.append(len(faces))

    # ── Graph 1: Frame grid with faces (IMPORTANT)
    n_show = min(6, len(extracted_frames))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        if i < n_show:
            frame  = extracted_frames[i]
            faces  = detect_faces(frame, 0.4)
            visual = frame.copy()
            for (x,y,fw,fh) in faces:
                cv2.rectangle(visual,(x,y),(x+fw,y+fh),(88,166,255),3)
            ax.imshow(visual)
            ax.set_title(f'Frame {i}  |  {len(faces)} face(s)', color='#c9d1d9', fontsize=9)
        ax.axis('off')
    plt.suptitle('Sampled Frame Analysis', color='#58a6ff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    R.save_graph('video_frames.png', 'Sampled Frames',
                 'Sampled frames from the video with face detection overlays.', important=True)
    plt.close(fig)

    # FFT per frame
    fft_scores = []
    for frame in extracted_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        fft_scores.append(float(np.std(np.log1p(np.abs(scipy_fftshift(scipy_fft2(gray)))))))

    # Temporal diff
    temporal_scores = []
    for i in range(len(extracted_frames)-1):
        g1 = cv2.cvtColor(extracted_frames[i],   cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(extracted_frames[i+1], cv2.COLOR_RGB2GRAY)
        temporal_scores.append(float(np.mean(cv2.absdiff(g1, g2))))

    # ── Graph 2: Temporal analysis (IMPORTANT)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(fft_scores, marker='o', color='#58a6ff', linewidth=2)
    axes[0].fill_between(range(len(fft_scores)), fft_scores, alpha=0.15, color='#58a6ff')
    axes[0].axhline(y=np.mean(fft_scores), color='#f85149', linestyle='--', label=f'Mean={np.mean(fft_scores):.2f}')
    axes[0].set_title('FFT Std Across Frames  — High consistency = suspicious', color='#c9d1d9')
    axes[0].set_xlabel('Frame Sample'); axes[0].legend(); axes[0].grid(True)

    if temporal_scores:
        axes[1].plot(temporal_scores, marker='o', color='#3fb950', linewidth=2)
        axes[1].fill_between(range(len(temporal_scores)), temporal_scores, alpha=0.15, color='#3fb950')
        axes[1].axhline(y=np.mean(temporal_scores), color='#f85149', linestyle='--', label=f'Mean={np.mean(temporal_scores):.2f}')
        axes[1].set_title('Temporal Difference  — Low values = unnatural stillness', color='#c9d1d9')
        axes[1].set_xlabel('Frame Pair'); axes[1].legend(); axes[1].grid(True)

    plt.suptitle('Temporal Forensic Analysis', color='#58a6ff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    R.save_graph('video_temporal.png', 'Temporal Analysis',
                 'FFT consistency and frame-to-frame differences. Deepfakes often show unnaturally consistent FFT scores.', important=True)
    plt.close(fig)

    R.add_stat('FFT Std Mean',      f'{np.mean(fft_scores):.2f}')
    R.add_stat('FFT Std Deviation', f'{np.std(fft_scores):.2f}')
    R.add_stat('Temporal Diff Mean', f'{np.mean(temporal_scores):.2f}' if temporal_scores else 'N/A')

    indicators  = []
    video_score = 0
    if np.std(fft_scores) < 2:
        video_score += 1; indicators.append('Unusually consistent FFT across frames')
    if temporal_scores and np.mean(temporal_scores) < 3:
        video_score += 1; indicators.append('Very low temporal variation between frames')
    if temporal_scores and np.std(temporal_scores) < 1:
        video_score += 1; indicators.append('Abnormally uniform frame-to-frame changes')
    if len(set(face_counts)) == 1 and face_counts[0] > 0:
        video_score += 1; indicators.append('Perfectly consistent face count across all frames')

    final_probability = min((video_score / 4) * 100, 100)

    for ind in indicators: R.add_indicator(f'[Video] {ind}')
    R.payload['stage_scores']['video_temporal'] = round(final_probability, 1)

    return final_probability
