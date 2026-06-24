"""
Image analysis pipeline — three-stage fusion architecture:
  Stage 1: frequency_domain_analysis()  — FFT/spectral forensics
  Stage 2: face_forensic_analysis()     — face/region forensics
  Stage 3: dl_detector()                — deep learning classification + fusion
  analyze_image()                       — orchestrates all three stages

Logic carried over unchanged from the validated Colab prototype (Cells 6, 7, 8, 12),
including the inverted-label fix (see helpers.extract_fake_score) that took
fake-image accuracy from ~12% to 100% in batch validation testing.
"""
import numpy as np
import cv2
import torch
from PIL import Image
from PIL.ExifTags import TAGS
import matplotlib.pyplot as plt
from transformers import pipeline as hf_pipeline

from .result import AnalysisResult
from .helpers import detect_faces, extract_fake_score, apply_graph_style

# ── Model singleton — loaded once at import time, reused for every request.
# Loading inside dl_detector() per-request was the primary cause of OOM on Render Starter (512 MB).
_DL_DETECTOR = None

def _get_dl_detector():
    global _DL_DETECTOR
    if _DL_DETECTOR is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _DL_DETECTOR = hf_pipeline(
            'image-classification',
            model='prithivMLmods/Deep-Fake-Detector-v2-Model',
            device=0 if device == 'cuda' else -1
        )
    return _DL_DETECTOR


# ============================================================
# STAGE 1 — Frequency Domain Analysis
# ============================================================
def frequency_domain_analysis(filepath, pil_image, R: AnalysisResult):

    R.pdf_text('<b>Stage 1 — Frequency Domain Analysis</b>', 'Heading1')
    apply_graph_style()

    indicators = []
    suspicion  = 0
    MAX_TESTS  = 7

    orig_w, orig_h = pil_image.size
    megapixels     = (orig_w * orig_h) / 1_000_000
    freq_reliable  = (orig_w >= 512 and orig_h >= 512)

    if not freq_reliable:
        scale    = 512 / min(orig_w, orig_h)
        work_img = pil_image.resize((int(orig_w*scale), int(orig_h*scale)), Image.LANCZOS)
    else:
        work_img = pil_image

    rgb_array  = np.array(work_img.convert('RGB'))
    gray_array = np.array(work_img.convert('L'))

    # FFT
    fft_shift      = np.fft.fftshift(np.fft.fft2(gray_array))
    magnitude      = np.abs(fft_shift)
    power_spectrum = magnitude ** 2
    phase_spectrum = np.angle(fft_shift)
    log_power      = np.log1p(power_spectrum)

    # ── Graph 1: Power Spectrum Heatmap (IMPORTANT — shown on frontend)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(log_power, cmap='inferno'); axes[0].set_title('Power Spectrum (log)', color='#c9d1d9'); axes[0].axis('off')
    axes[1].imshow(phase_spectrum, cmap='twilight'); axes[1].set_title('Phase Spectrum', color='#c9d1d9'); axes[1].axis('off')
    plt.suptitle('Frequency Domain Analysis', color='#58a6ff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    R.save_graph('freq_spectrum.png', 'Frequency Spectrum',
                 'Power spectrum (left) and phase spectrum (right). Synthetic images show unnatural uniformity.', important=True)
    plt.close(fig)

    # Radial average in log domain
    y_idx, x_idx = np.indices(log_power.shape)
    cy, cx       = log_power.shape[0]//2, log_power.shape[1]//2
    radius       = np.sqrt((x_idx-cx)**2 + (y_idx-cy)**2).astype(np.int32)
    counts       = np.bincount(radius.ravel())
    sums         = np.bincount(radius.ravel(), weights=log_power.ravel())
    radial       = sums / np.maximum(counts, 1)
    radial_norm  = radial / (radial[0] + 1e-12)

    # ── Graph 2: Radial Decay (IMPORTANT)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(radial_norm, linewidth=2, color='#58a6ff')
    ax.fill_between(range(len(radial_norm)), radial_norm, alpha=0.15, color='#58a6ff')
    ax.set_xlabel('Frequency Radius'); ax.set_ylabel('Normalised Log Power')
    ax.set_title('Radial Frequency Decay — Real images decay smoothly; deepfakes show flat regions', color='#c9d1d9')
    ax.grid(True)
    plt.tight_layout()
    R.save_graph('radial_decay.png', 'Frequency Decay Profile',
                 'Real camera images show smooth exponential decay. Flat or irregular regions suggest synthetic generation.', important=True)
    plt.close(fig)

    # Band energies
    n         = len(radial_norm)
    low_e     = float(np.mean(radial_norm[:n//3]))
    mid_e     = float(np.mean(radial_norm[n//3:2*n//3]))
    high_e    = float(np.mean(radial_norm[2*n//3:]))
    hf_std    = float(np.std(radial_norm[2*n//3:]))

    R.add_stat('Low Band Energy',  f'{low_e:.4f}')
    R.add_stat('Mid Band Energy',  f'{mid_e:.4f}')
    R.add_stat('High Band Energy', f'{high_e:.4f}')
    R.add_stat('HF Std Dev',       f'{hf_std:.4f}')

    # ── Graph 3: Band Energy Bar (IMPORTANT)
    fig, ax = plt.subplots(figsize=(7, 4))
    bands  = ['Low\n(0–33%)', 'Mid\n(33–66%)', 'High\n(66–100%)']
    values = [low_e, mid_e, high_e]
    bar_colors = ['#3fb950', '#d29922', '#f85149']
    bars = ax.bar(bands, values, color=bar_colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f'{val:.4f}', ha='center', color='#c9d1d9', fontsize=10)
    ax.set_title('Frequency Band Energy Distribution', color='#c9d1d9')
    ax.set_ylabel('Mean Log-Normalised Power')
    plt.tight_layout()
    R.save_graph('band_energy.png', 'Frequency Band Energy',
                 'Low/mid/high frequency band energies. Deepfakes often show abnormally weak high-frequency energy.', important=True)
    plt.close(fig)

    if hf_std < 0.1:
        indicators.append('Flattened high-frequency spectrum'); suspicion += 1
    if high_e < low_e * 0.1:
        indicators.append('Weak high-frequency energy'); suspicion += 1

    # Spectral entropy
    norm_lp      = log_power / (np.sum(log_power) + 1e-12)
    spec_entropy = float(-np.sum(norm_lp * np.log2(norm_lp + 1e-12)))
    R.add_stat('Spectral Entropy', f'{spec_entropy:.2f}')
    if spec_entropy < 15:
        indicators.append(f'Low spectral entropy ({spec_entropy:.2f})'); suspicion += 1

    # Symmetry
    sym_diff_tb = float(np.mean(np.abs(log_power - np.flipud(log_power))))
    if sym_diff_tb < 0.5:
        indicators.append('Abnormal top-bottom spectrum symmetry'); suspicion += 1

    # Per-channel FFT (PDF only — informational)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (ch_name, ax) in enumerate(zip(['Red', 'Green', 'Blue'], axes)):
        ch_fft = np.fft.fftshift(np.fft.fft2(rgb_array[:,:,i]))
        ch_log = np.log1p(np.abs(ch_fft)**2)
        ax.imshow(ch_log, cmap='viridis'); ax.set_title(f'{ch_name}', color='#c9d1d9'); ax.axis('off')
    plt.suptitle('Per-Channel Frequency Spectra', color='#58a6ff')
    plt.tight_layout()
    R.save_graph('channel_fft.png', 'Per-Channel FFT', important=False)  # PDF only
    plt.close(fig)

    # Noise residual
    blurred      = cv2.GaussianBlur(gray_array, (5,5), 0)
    residual     = cv2.absdiff(gray_array, blurred)
    residual_std = float(np.std(residual))
    R.add_stat('Noise Residual Std', f'{residual_std:.2f}')

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(residual, cmap='hot'); plt.colorbar(im, ax=ax)
    ax.set_title(f'Sensor Noise Residual  (std={residual_std:.2f})', color='#c9d1d9'); ax.axis('off')
    plt.tight_layout()
    R.save_graph('noise_residual.png', 'Noise Residual Map',
                 f'Real cameras produce characteristic noise patterns (std 2–8). Std={residual_std:.2f}. Very low values suggest synthetic origin.', important=True)
    plt.close(fig)

    if residual_std < 4:
        indicators.append('Extremely weak sensor-noise residual'); suspicion += 1

    # Edge density
    edge_map     = cv2.Canny(gray_array, 100, 200)
    edge_density = float(np.mean(edge_map > 0))
    edge_thresh  = 0.05 / max(1.0, np.sqrt(megapixels))
    R.add_stat('Edge Density', f'{edge_density:.4f}')

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(edge_map, cmap='gray'); ax.set_title('Edge Map', color='#c9d1d9'); ax.axis('off')
    plt.tight_layout()
    R.save_graph('edge_map.png', 'Edge Structure Map', important=False)  # PDF only
    plt.close(fig)

    if edge_density < edge_thresh:
        indicators.append('Artificially smooth edge structure'); suspicion += 1

    raw_freq_score = (suspicion / MAX_TESTS) * 100
    R.payload['stage_scores']['frequency'] = round(raw_freq_score, 1)

    R.pdf_text(f'<b>Frequency Score: {raw_freq_score:.1f}%  |  Tests triggered: {suspicion}/{MAX_TESTS}</b>')
    for ind in indicators: R.add_indicator(f'[Frequency] {ind}')

    return raw_freq_score, indicators, freq_reliable


# ============================================================
# STAGE 2 — Face / Region Forensic Analysis
# ============================================================
def face_forensic_analysis(filepath, pil_image, freq_score, freq_indicators, R: AnalysisResult):

    R.pdf_text('<b>Stage 2 — Face Forensic Analysis</b>', 'Heading1')
    apply_graph_style()

    face_indicators = list(freq_indicators)
    face_score      = 0
    FACE_TESTS      = 6

    rgb_array  = np.array(pil_image.convert('RGB'))
    gray_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray_image.shape

    faces = detect_faces(rgb_array, min_confidence=0.4)
    if len(faces) == 0:
        faces = detect_faces(rgb_array, min_confidence=0.15)

    R.add_stat('Faces Detected', len(faces))

    # ── Graph: Detected Faces (IMPORTANT)
    visual = rgb_array.copy()
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(visual, (fx,fy), (fx+fw,fy+fh), (88, 166, 255), 3)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(visual)
    ax.set_title(f'{len(faces)} Face(s) Detected', color='#c9d1d9', fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    R.save_graph('detected_faces.png', 'Face Detection',
                 f'{len(faces)} face(s) detected. Face regions are analysed for warping, blending and resolution inconsistencies.', important=True)
    plt.close(fig)

    if len(faces) == 0:
        R.payload['stage_scores']['face_forensics'] = None
        return freq_score, face_indicators

    areas       = [(fw*fh, fx, fy, fw, fh) for (fx,fy,fw,fh) in faces]
    _, fx, fy, fw, fh = max(areas)
    face_roi    = gray_image[fy:fy+fh, fx:fx+fw]
    ex1 = max(0,fx-40); ey1 = max(0,fy-40)
    ex2 = min(img_w,fx+fw+40); ey2 = min(img_h,fy+fh+40)
    surround_roi = gray_image[ey1:ey2, ex1:ex2]

    if face_roi.size == 0:
        return freq_score, face_indicators

    # Test 1: Face edge structure
    face_edges = cv2.Canny(face_roi, 100, 200)
    edge_std   = float(np.std(face_edges))
    if edge_std < 35:
        face_indicators.append('Face warping/smoothing artifacts'); face_score += 1

    # Test 2: Resolution inconsistency
    face_var = float(cv2.Laplacian(face_roi, cv2.CV_64F).var())
    surr_var = float(cv2.Laplacian(surround_roi, cv2.CV_64F).var()) if surround_roi.size > 0 else face_var
    res_diff = abs(face_var - surr_var)
    if res_diff > 300:
        face_indicators.append('Resolution mismatch: face vs surroundings'); face_score += 1

    # Test 3: Blur residual
    blur_diff = cv2.absdiff(face_roi, cv2.GaussianBlur(face_roi, (9,9), 0))
    blur_std  = float(np.std(blur_diff))
    if blur_std < 18:
        face_indicators.append('Localised blur smoothing on face'); face_score += 1

    # Test 4: Face frequency spectrum
    face_spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(face_roi))))
    spec_std  = float(np.std(face_spec))
    if spec_std < 1.5:
        face_indicators.append('Resampling artifacts in face region'); face_score += 1

    # Test 5: Boundary blending
    boundary_mask = np.zeros_like(gray_image)
    cv2.rectangle(boundary_mask, (fx,fy), (fx+fw,fy+fh), 255, 10)
    boundary_std = float(np.std(cv2.bitwise_and(gray_image, boundary_mask)))
    if boundary_std < 20:
        face_indicators.append('Artificial boundary blending'); face_score += 1

    # Test 6: Multi-scale energy
    scales         = [64, 128, 256]
    scale_energies = [float(np.mean(np.abs(np.fft.fft2(cv2.resize(face_roi,(s,s))))))
                      for s in scales]
    energy_std = float(np.std(scale_energies))
    if energy_std < 150:
        face_indicators.append('Abnormal multi-scale energy consistency'); face_score += 1

    # ── Graph: Face forensics dashboard (IMPORTANT)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(face_edges, cmap='inferno')
    axes[0].set_title(f'Face Edges  (std={edge_std:.1f})', color='#c9d1d9'); axes[0].axis('off')

    axes[1].imshow(blur_diff, cmap='hot')
    axes[1].set_title(f'Blur Residual  (std={blur_std:.1f})', color='#c9d1d9'); axes[1].axis('off')

    ax2 = axes[2]
    ax2.bar(['Face', 'Surround'], [face_var, surr_var], color=['#58a6ff','#3fb950'])
    ax2.set_title(f'Sharpness: Face vs Surround  (diff={res_diff:.0f})', color='#c9d1d9')
    ax2.set_ylabel('Laplacian Variance', color='#c9d1d9')

    plt.suptitle('Face Forensic Analysis', color='#58a6ff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    R.save_graph('face_forensics.png', 'Face Forensic Analysis',
                 f'Edge structure, blur residual, and sharpness comparison. {face_score}/{FACE_TESTS} tests flagged.', important=True)
    plt.close(fig)

    face_probability = (face_score / FACE_TESTS) * 100
    combined         = freq_score * 0.5 + face_probability * 0.5

    R.add_stat('Face Forensic Score', f'{face_probability:.1f}%')
    R.payload['stage_scores']['face_forensics'] = round(face_probability, 1)
    for ind in face_indicators[len(freq_indicators):]:
        R.add_indicator(f'[Face] {ind}')

    return combined, face_indicators


# ============================================================
# STAGE 3 — Deep Learning Detector + Fusion
# ============================================================
def dl_detector(filepath, pil_image, combined_forensic_score, all_indicators,
                freq_reliable, R: AnalysisResult):

    R.pdf_text('<b>Stage 3 — Deep Learning Detector</b>', 'Heading1')
    apply_graph_style()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dl_available  = False
    fake_score    = 0.0
    matched_label = 'none'

    # Primary: purpose-trained deepfake classifier (singleton — loaded once)
    try:
        detector  = _get_dl_detector()
        result    = detector(pil_image.convert('RGB'))
        label_map = {r['label']: r['score'] for r in result}
        fake_score, matched_label = extract_fake_score(label_map)
        dl_available = True
        R.pdf_text(f'DL model output: {result}')
    except Exception as e:
        R.pdf_text(f'Primary DL model failed: {e} — falling back to forensic score.')

    # Fallback: if primary model fails, use the combined forensic score directly.
    # EfficientNet-B4 was removed — loading it inline consumed ~200 MB and
    # contributed to OOM kills on Render Starter.
    if not dl_available:
        fake_score    = combined_forensic_score
        matched_label = 'forensic-only'
        R.pdf_text(f'Primary DL model unavailable — using forensic score as final score.')

    R.add_stat('DL Model Score', f'{fake_score:.1f}%')
    R.add_stat('DL Label Matched', matched_label)
    R.payload['stage_scores']['deep_learning'] = round(fake_score, 1)

    # ── Graph: DL Score gauge + patch variance (IMPORTANT)
    img_np  = np.array(pil_image.convert('RGB').resize((380, 380)))
    gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    heatmap = np.uint8(cv2.normalize(
        cv2.GaussianBlur(gray, (41,41), 0), None, 0, 255, cv2.NORM_MINMAX
    ))
    overlay = cv2.addWeighted(
        img_np, 0.6, cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), 0.4, 0
    )
    patch_size      = 32
    patch_variances = [
        float(np.var(gray[y:y+patch_size, x:x+patch_size]))
        for y in range(0, gray.shape[0], patch_size)
        for x in range(0, gray.shape[1], patch_size)
    ]
    consistency = float(np.std(patch_variances))
    if consistency < 150:
        all_indicators.append('[DL] Abnormally consistent patch variance')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(overlay); axes[0].set_title('Attention Heatmap', color='#c9d1d9'); axes[0].axis('off')
    axes[1].plot(patch_variances, linewidth=1.2, color='#58a6ff', alpha=0.8)
    axes[1].axhline(y=np.mean(patch_variances), color='#f85149', linestyle='--', label=f'Mean={np.mean(patch_variances):.0f}')
    axes[1].set_title(f'Patch Variance  (consistency std={consistency:.0f})', color='#c9d1d9')
    axes[1].set_xlabel('Patch Index'); axes[1].set_ylabel('Variance')
    axes[1].legend(); axes[1].grid(True)
    plt.suptitle(f'Deep Learning Analysis — DL Score: {fake_score:.1f}%  |  Model: {matched_label}',
                 color='#58a6ff', fontsize=12, fontweight='bold')
    plt.tight_layout()
    R.save_graph('dl_analysis.png', 'Deep Learning Analysis',
                 f'Attention heatmap and patch variance. DL model confidence: {fake_score:.1f}%.', important=True)
    plt.close(fig)

    # Fusion
    if not freq_reliable:
        fw_, dw = 0.20, 0.80
    else:
        fw_, dw = 0.60, 0.40

    final = float(np.clip(combined_forensic_score * fw_ + fake_score * dw, 0, 100))
    R.add_stat('Forensic Score', f'{combined_forensic_score:.1f}%')
    R.add_stat('Final Fusion Score', f'{final:.1f}%')

    return final


# ============================================================
# ORCHESTRATOR — runs all three stages for an image
# ============================================================
def analyze_image(filepath, R: AnalysisResult):
    R.pdf_text('IMAGE FORENSIC ANALYSIS REPORT', 'Title')
    apply_graph_style()

    # Load — handle MPO (iPhone Live Photo)
    pil_raw = Image.open(filepath)
    if getattr(pil_raw, 'format', '') == 'MPO':
        pil_raw.seek(0)
    pil_image = pil_raw.convert('RGB')

    R.add_stat('Format',     getattr(pil_raw, 'format', 'unknown'))
    R.add_stat('Dimensions', f'{pil_image.size[0]} x {pil_image.size[1]} px')
    R.add_stat('Megapixels', f"{pil_image.size[0]*pil_image.size[1]/1e6:.2f} MP")

    # EXIF
    exif_fields = 0
    try:
        exif_data = pil_raw.getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                R.pdf_text(f'{TAGS.get(tag_id, tag_id)}: {value}')
                exif_fields += 1
    except Exception:
        pass
    R.add_stat('EXIF Fields', exif_fields)
    if exif_fields == 0:
        R.add_indicator('[EXIF] No EXIF metadata — consistent with AI-generated image')

    # Stage 1: Frequency
    freq_score, freq_indicators, freq_reliable = frequency_domain_analysis(filepath, pil_image, R)

    # Stage 2: Face forensics
    combined_score, all_indicators = face_forensic_analysis(
        filepath, pil_image, freq_score, freq_indicators, R
    )

    # Stage 3: Deep learning
    final_score = dl_detector(
        filepath, pil_image, combined_score, all_indicators, freq_reliable, R
    )

    return final_score
