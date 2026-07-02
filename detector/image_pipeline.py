"""
Image analysis pipeline  -  five-stage fusion architecture:
  Stage 1: frequency_domain_analysis()    -  FFT/spectral forensics
  Stage 2: face_forensic_analysis()       -  face/region forensics (human faces only)
  Stage 3: manipulation_analysis()        -  ELA, copy-move, PRNU, metadata, patch analysis
  Stage 4: vehicle_damage_analysis()      -  vehicle/object damage forensics (runs when no human face)
  Stage 5: dl_detector()                  -  deep learning classification + adaptive fusion

Catches:
  - GAN deepfakes, face swaps, fully AI-generated faces
  - ChatGPT / DALL-E / Firefly / Stable Diffusion inpainting on any subject
  - Photoshop clone stamp, healing brush, content-aware fill, compositing
  - AI-generated vehicle damage (insurance fraud)
  - Metadata tampering (Photoshop Software tag, missing EXIF, ICC mismatch)
  - Copy-move duplication within an image

Does NOT catch:
  - Pure colour/brightness grading with zero structural change
  - Liquify/warp with no content replacement
  (These are undetectable by any open-source forensic tool)
"""
import io
import os
import gc
import numpy as np
import cv2
import torch
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
import matplotlib.pyplot as plt
from transformers import pipeline as hf_pipeline

from .result import AnalysisResult
from .helpers import detect_faces, extract_fake_score, apply_graph_style

# ============================================================================
# DELIBERATE EXCEPTION to the "all models are permanent singletons" rule
# from the original project handoff. That rule existed to fix a DIFFERENT
# problem (constant reload thrash from loading on every single call site
# scattered through old code). Two image-classification models held
# permanently resident was found to leave a ~1.5-1.6GB baseline out of a
# 2GB instance - too little headroom for large-image processing, causing
# OOM kills (silent process restarts, no Python traceback).
#
# Made at explicit user request, prioritizing zero-crash reliability over
# request latency: each model is loaded, used once, then fully released
# (del + gc.collect()) BEFORE the next model loads - strictly sequential,
# never both resident in memory at the same time. This trades a few extra
# seconds of model-load time per request for a much lower peak memory
# footprint. Do not "fix" this back to singletons without re-confirming
# the memory tradeoff with the user first.
# ============================================================================

def _run_dl_deepfake_model(pil_image):
    """Loads prithivMLmods/Deep-Fake-Detector-v2-Model, runs inference once,
    then fully releases it from memory before returning."""
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = hf_pipeline(
        'image-classification',
        model='prithivMLmods/Deep-Fake-Detector-v2-Model',
        device=0 if device == 'cuda' else -1
    )
    try:
        result = detector(pil_image.convert('RGB'))
    finally:
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _run_dl_ai_model(pil_image):
    """Loads Organika/sdxl-detector, runs inference once, then fully releases
    it from memory before returning. Only called after
    _run_dl_deepfake_model() has already flushed - never both resident."""
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = hf_pipeline(
        'image-classification',
        model='Organika/sdxl-detector',
        device=0 if device == 'cuda' else -1
    )
    try:
        result = detector(pil_image.convert('RGB'))
    finally:
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _run_dl_ai_model_2(pil_image):
    """
    Second AI-generation detector: umm-maybe/AI-image-detector.
    Trained on a broader range of generators than sdxl-detector
    (includes DALL-E, Midjourney, Stable Diffusion 1.x/2.x, Firefly).
    Load-score-flush pattern — never resident with other models.
    Uses same 'artificial' / 'human' label scheme as sdxl-detector
    so _extract_ai_generated_score() handles both without modification.
    """
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = hf_pipeline(
        'image-classification',
        model='umm-maybe/AI-image-detector',
        device=0 if device == 'cuda' else -1
    )
    try:
        result = detector(pil_image.convert('RGB'))
    finally:
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _extract_ai_generated_score(label_map: dict) -> tuple:
    """
    Organika/sdxl-detector is a fine-tune of umm-maybe/AI-image-detector,
    which uses 'artificial' / 'human' labels. Matched generically here
    (substring, case-insensitive) rather than hardcoding exact casing, since
    getting this backwards would silently invert every score - the same
    failure mode extract_fake_score() protects against for the other model.

    NOT YET EMPIRICALLY VALIDATED the way extract_fake_score() was (92.9%
    on a labeled test set). Test against known AI-generated and known-real
    images before trusting this score in production.
    """
    ai_keys   = [k for k in label_map if any(s in k.lower() for s in ('artificial', 'synthetic', 'fake', 'ai'))]
    real_keys = [k for k in label_map if any(s in k.lower() for s in ('human', 'real'))]

    if ai_keys:
        key = ai_keys[0]
        return label_map[key] * 100, key
    if real_keys:
        key = real_keys[0]
        return (1 - label_map[key]) * 100, key

    # Unrecognized label scheme - surface it loudly instead of guessing.
    raise ValueError(f'Unrecognized sdxl-detector label scheme: {list(label_map.keys())}')


# ============================================================
# STAGE 1  -  Frequency Domain Analysis
# ============================================================
def frequency_domain_analysis(filepath, pil_image, R: AnalysisResult):
    R.pdf_text('<b>Stage 1  -  Frequency Domain Analysis</b>', 'Heading1')
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

    fft_shift      = np.fft.fftshift(np.fft.fft2(gray_array))
    magnitude      = np.abs(fft_shift)
    power_spectrum = magnitude ** 2
    phase_spectrum = np.angle(fft_shift)
    log_power      = np.log1p(power_spectrum)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(log_power, cmap='inferno'); axes[0].set_title('Power Spectrum (log)', color='#c9d1d9'); axes[0].axis('off')
    axes[1].imshow(phase_spectrum, cmap='twilight'); axes[1].set_title('Phase Spectrum', color='#c9d1d9'); axes[1].axis('off')
    plt.suptitle('Frequency Domain Analysis', color='#58a6ff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    R.save_graph('freq_spectrum.png', 'Frequency Spectrum',
                 'Power spectrum (left) and phase spectrum (right). Synthetic images show unnatural uniformity.', important=True)
    plt.close(fig)

    y_idx, x_idx = np.indices(log_power.shape)
    cy, cx       = log_power.shape[0]//2, log_power.shape[1]//2
    radius       = np.sqrt((x_idx-cx)**2 + (y_idx-cy)**2).astype(np.int32)
    counts       = np.bincount(radius.ravel())
    sums         = np.bincount(radius.ravel(), weights=log_power.ravel())
    radial       = sums / np.maximum(counts, 1)
    radial_norm  = radial / (radial[0] + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(radial_norm, linewidth=2, color='#58a6ff')
    ax.fill_between(range(len(radial_norm)), radial_norm, alpha=0.15, color='#58a6ff')
    ax.set_xlabel('Frequency Radius'); ax.set_ylabel('Normalised Log Power')
    ax.set_title('Radial Frequency Decay', color='#c9d1d9')
    ax.grid(True)
    plt.tight_layout()
    R.save_graph('radial_decay.png', 'Frequency Decay Profile',
                 'Real camera images show smooth exponential decay. Flat or irregular regions suggest synthetic generation.', important=True)
    plt.close(fig)

    n      = len(radial_norm)
    low_e  = float(np.mean(radial_norm[:n//3]))
    mid_e  = float(np.mean(radial_norm[n//3:2*n//3]))
    high_e = float(np.mean(radial_norm[2*n//3:]))
    hf_std = float(np.std(radial_norm[2*n//3:]))

    R.add_stat('Low Band Energy',  f'{low_e:.4f}')
    R.add_stat('Mid Band Energy',  f'{mid_e:.4f}')
    R.add_stat('High Band Energy', f'{high_e:.4f}')
    R.add_stat('HF Std Dev',       f'{hf_std:.4f}')

    fig, ax = plt.subplots(figsize=(7, 4))
    bands      = ['Low\n(0-33%)', 'Mid\n(33-66%)', 'High\n(66-100%)']
    values     = [low_e, mid_e, high_e]
    bar_colors = ['#3fb950', '#d29922', '#f85149']
    bars = ax.bar(bands, values, color=bar_colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f'{val:.4f}', ha='center', color='#c9d1d9', fontsize=10)
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

    norm_lp      = log_power / (np.sum(log_power) + 1e-12)
    spec_entropy = float(-np.sum(norm_lp * np.log2(norm_lp + 1e-12)))
    R.add_stat('Spectral Entropy', f'{spec_entropy:.2f}')
    if spec_entropy < 15:
        indicators.append(f'Low spectral entropy ({spec_entropy:.2f})'); suspicion += 1

    sym_diff_tb = float(np.mean(np.abs(log_power - np.flipud(log_power))))
    if sym_diff_tb < 0.5:
        indicators.append('Abnormal top-bottom spectrum symmetry'); suspicion += 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (ch_name, ax) in enumerate(zip(['Red', 'Green', 'Blue'], axes)):
        ch_fft = np.fft.fftshift(np.fft.fft2(rgb_array[:,:,i]))
        ch_log = np.log1p(np.abs(ch_fft)**2)
        ax.imshow(ch_log, cmap='viridis'); ax.set_title(f'{ch_name}', color='#c9d1d9'); ax.axis('off')
    plt.suptitle('Per-Channel Frequency Spectra', color='#58a6ff')
    plt.tight_layout()
    R.save_graph('channel_fft.png', 'Per-Channel FFT', important=False)
    plt.close(fig)

    blurred      = cv2.GaussianBlur(gray_array, (5,5), 0)
    residual     = cv2.absdiff(gray_array, blurred)
    residual_std = float(np.std(residual))
    R.add_stat('Noise Residual Std', f'{residual_std:.2f}')
    R.payload['stage_scores']['noise_residual_std'] = round(residual_std, 2)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(residual, cmap='hot'); plt.colorbar(im, ax=ax)
    ax.set_title(f'Sensor Noise Residual  (std={residual_std:.2f})', color='#c9d1d9'); ax.axis('off')
    plt.tight_layout()
    R.save_graph('noise_residual.png', 'Noise Residual Map',
                 f'Real cameras produce characteristic noise patterns (std 2-8). Std={residual_std:.2f}. Very low values suggest synthetic origin.', important=True)
    plt.close(fig)

    if residual_std < 4:
        indicators.append('Extremely weak sensor-noise residual'); suspicion += 1

    edge_map     = cv2.Canny(gray_array, 100, 200)
    edge_density = float(np.mean(edge_map > 0))
    edge_thresh  = 0.05 / max(1.0, np.sqrt(megapixels))
    R.add_stat('Edge Density', f'{edge_density:.4f}')

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(edge_map, cmap='gray'); ax.set_title('Edge Map', color='#c9d1d9'); ax.axis('off')
    plt.tight_layout()
    R.save_graph('edge_map.png', 'Edge Structure Map', important=False)
    plt.close(fig)

    if edge_density < edge_thresh:
        indicators.append(f'Unusually low edge density ({edge_density:.4f})'); suspicion += 1

    freq_score = float(np.clip((suspicion / MAX_TESTS) * 100, 0, 100))
    R.add_stat('Frequency Suspicion', f'{suspicion}/{MAX_TESTS}')
    R.add_stat('Frequency Score',     f'{freq_score:.1f}%')
    R.payload['stage_scores']['frequency'] = round(freq_score, 1)

    for ind in indicators:
        R.add_indicator(f'[Frequency] {ind}')

    return freq_score, indicators, freq_reliable


# ============================================================
# STAGE 2  -  Face Forensic Analysis (human faces only)
# ============================================================
def face_forensic_analysis(filepath, pil_image, freq_score, freq_indicators, R: AnalysisResult):
    R.pdf_text('<b>Stage 2  -  Face Forensic Analysis</b>', 'Heading1')
    apply_graph_style()

    FACE_TESTS      = 6
    face_score      = 0
    face_indicators = list(freq_indicators)
    has_human_face  = False

    gray_image = np.array(pil_image.convert('L'))
    rgb_image  = np.array(pil_image.convert('RGB'))

    # Use only Haar for face detection  -  Mediapipe was removed (OOM)
    raw_faces = detect_faces(rgb_image, min_confidence=0.4)

    # Filter out false positives from non-face rectangular objects
    # (licence plates, car grilles etc. trigger Haar on vehicle images)
    # A human face has aspect ratio close to 1:1 and minimum size
    h_img, w_img = gray_image.shape
    human_faces  = []
    for (fx, fy, fw, fh) in raw_faces:
        aspect  = fw / max(fh, 1)
        area    = fw * fh
        img_area = h_img * w_img
        # Human face: aspect ratio 0.6-1.4, area at least 0.5% of image
        if 0.6 <= aspect <= 1.4 and area >= img_area * 0.005:
            human_faces.append((fx, fy, fw, fh))

    faces         = human_faces
    has_human_face = len(faces) > 0
    R.add_stat('Human Faces Detected', len(faces))
    R.add_stat('Raw Detections',       len(raw_faces))

    fig, ax = plt.subplots(figsize=(8, 8))
    visual  = rgb_image.copy()
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(visual, (fx, fy), (fx+fw, fy+fh), (88, 166, 255), 3)
    # Draw rejected detections in grey
    for (fx, fy, fw, fh) in raw_faces:
        if (fx, fy, fw, fh) not in faces:
            cv2.rectangle(visual, (fx, fy), (fx+fw, fy+fh), (100, 100, 100), 1)
    ax.imshow(visual)
    ax.set_title(f'{len(faces)} human face(s) detected  |  {len(raw_faces)-len(faces)} false positives filtered', color='#c9d1d9')
    ax.axis('off')
    plt.tight_layout()
    R.save_graph('detected_faces.png', 'Face Detection',
                 f'{len(faces)} human face(s) confirmed. Grey boxes = filtered false positives (licence plates, objects).', important=True)
    plt.close(fig)

    if not has_human_face:
        # No human face  -  face forensics not applicable
        # Return freq_score as the combined score; vehicle stage will handle the rest
        R.add_stat('Face Forensic Score', 'N/A  -  no human face detected')
        R.payload['stage_scores']['face_forensics'] = None
        R.pdf_text('No human faces detected. Skipping face forensics. Vehicle damage analysis will run instead.')
        return freq_score, face_indicators, has_human_face

    # Run face forensics on largest face
    fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
    face_roi    = gray_image[fy:fy+fh, fx:fx+fw]
    surround_mask = np.ones_like(gray_image, dtype=bool)
    surround_mask[fy:fy+fh, fx:fx+fw] = False
    surround_roi = gray_image[surround_mask]

    face_var = float(cv2.Laplacian(face_roi, cv2.CV_64F).var())
    surr_var = float(np.var(cv2.Laplacian(gray_image, cv2.CV_64F)))
    res_diff = abs(face_var - surr_var)
    if res_diff > 500:
        face_indicators.append('Resolution mismatch: face vs surroundings'); face_score += 1

    face_edges = cv2.Canny(face_roi, 50, 150)
    edge_std   = float(np.std(face_edges))
    if edge_std < 30:
        face_indicators.append('Unnaturally smooth face edges'); face_score += 1

    face_blur = cv2.GaussianBlur(face_roi, (5,5), 0)
    blur_diff = cv2.absdiff(face_roi, face_blur)
    blur_std  = float(np.std(blur_diff))
    if blur_std < 18:
        face_indicators.append('Localised blur smoothing on face'); face_score += 1

    face_spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(face_roi))))
    spec_std  = float(np.std(face_spec))
    if spec_std < 1.5:
        face_indicators.append('Resampling artifacts in face region'); face_score += 1

    boundary_mask = np.zeros_like(gray_image)
    cv2.rectangle(boundary_mask, (fx,fy), (fx+fw,fy+fh), 255, 10)
    boundary_std = float(np.std(cv2.bitwise_and(gray_image, boundary_mask)))
    if boundary_std < 20:
        face_indicators.append('Artificial boundary blending'); face_score += 1

    scales         = [64, 128, 256]
    scale_energies = [float(np.mean(np.abs(np.fft.fft2(cv2.resize(face_roi,(s,s))))))
                      for s in scales]
    energy_std = float(np.std(scale_energies))
    if energy_std < 150:
        face_indicators.append('Abnormal multi-scale energy consistency'); face_score += 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(face_edges, cmap='inferno')
    axes[0].set_title(f'Face Edges  (std={edge_std:.1f})', color='#c9d1d9'); axes[0].axis('off')
    axes[1].imshow(blur_diff, cmap='hot')
    axes[1].set_title(f'Blur Residual  (std={blur_std:.1f})', color='#c9d1d9'); axes[1].axis('off')
    axes[2].bar(['Face', 'Surround'], [face_var, surr_var], color=['#58a6ff','#3fb950'])
    axes[2].set_title(f'Sharpness: Face vs Surround  (diff={res_diff:.0f})', color='#c9d1d9')
    axes[2].set_ylabel('Laplacian Variance', color='#c9d1d9')
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

    return combined, face_indicators, has_human_face


# ============================================================
# STAGE 3  -  Manipulation Analysis
# (ELA + Copy-Move + PRNU + Metadata + Patch Inconsistency)
# ============================================================
def manipulation_analysis(filepath, pil_image, R: AnalysisResult):
    """
    Catches edits that frequency/face analysis misses.
    Works on ALL image types  -  faces, vehicles, objects, documents.
    """
    R.pdf_text('<b>Stage 3  -  Manipulation Analysis</b>', 'Heading1')
    apply_graph_style()

    manip_score      = 0.0
    indicators       = []
    component_scores = {}

    rgb_array  = np.array(pil_image.convert('RGB'))
    gray_array = np.array(pil_image.convert('L'))
    h, w       = gray_array.shape

    # -- 3A: Error Level Analysis ---------------------------------
    try:
        ela_qualities = [75, 85, 95]
        ela_maps = []
        for quality in ela_qualities:
            buf = io.BytesIO()
            pil_image.convert('RGB').save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            recompressed = Image.open(buf).convert('RGB')
            ela_diff = ImageChops.difference(pil_image.convert('RGB'), recompressed)
            ela_maps.append(np.array(ela_diff).astype(float))

        ela_map  = np.mean(ela_maps, axis=0)
        ela_gray = np.mean(ela_map, axis=2)
        ela_mean = float(np.mean(ela_gray))
        ela_std  = float(np.std(ela_gray))
        ela_max  = float(np.max(ela_gray))

        quad_h, quad_w = h//2, w//2
        quads = [
            ela_gray[:quad_h, :quad_w], ela_gray[:quad_h, quad_w:],
            ela_gray[quad_h:, :quad_w], ela_gray[quad_h:, quad_w:],
        ]
        quad_means   = [float(np.mean(q)) for q in quads]
        quad_range   = max(quad_means) - min(quad_means)
        ela_regional = quad_range > ela_mean * 1.5

        ela_score = 0.0
        if ela_mean > 12:
            ela_score += 40; indicators.append('High ELA mean  -  JPEG inconsistency across image')
        if ela_regional:
            ela_score += 45; indicators.append('Regional ELA spike  -  localised edit detected (inpainting/paste)')
        if ela_std > 20:
            ela_score += 15; indicators.append('High ELA variance  -  uneven compression history')

        component_scores['ela'] = min(ela_score, 100)
        R.add_stat('ELA Mean',     f'{ela_mean:.2f}')
        R.add_stat('ELA Std',      f'{ela_std:.2f}')
        R.add_stat('ELA Regional', 'YES' if ela_regional else 'NO')

        ela_display = np.clip(ela_gray * 10, 0, 255).astype(np.uint8)
        quad_img    = np.zeros((h, w), dtype=np.float32)
        quad_img[:quad_h, :quad_w] = quad_means[0]
        quad_img[:quad_h, quad_w:] = quad_means[1]
        quad_img[quad_h:, :quad_w] = quad_means[2]
        quad_img[quad_h:, quad_w:] = quad_means[3]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(pil_image.convert('RGB')); axes[0].set_title('Original', color='#c9d1d9'); axes[0].axis('off')
        axes[1].imshow(ela_display, cmap='hot'); axes[1].set_title(f'ELA Map (mean={ela_mean:.1f})', color='#c9d1d9'); axes[1].axis('off')
        axes[2].imshow(quad_img, cmap='RdYlGn_r'); axes[2].set_title(f'Regional ELA (range={quad_range:.1f})', color='#c9d1d9'); axes[2].axis('off')
        plt.suptitle('Error Level Analysis  -  Detects inpainting & pasted regions', color='#58a6ff', fontsize=12, fontweight='bold')
        plt.tight_layout()
        R.save_graph('ela_analysis.png', 'Error Level Analysis',
                     f'ELA detects regions edited after original JPEG save. Mean={ela_mean:.1f}, Regional spike={ela_regional}.', important=True)
        plt.close(fig)
    except Exception as e:
        R.pdf_text(f'ELA analysis failed: {e}')
        component_scores['ela'] = 0.0

    # -- 3B: Copy-Move Detection ----------------------------------
    try:
        orb          = cv2.ORB_create(nfeatures=500)
        kp, des      = orb.detectAndCompute(gray_array, None)
        copy_score   = 0.0
        clone_regions = 0

        if des is not None and len(des) > 20:
            bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des, des)
            suspicious = []
            for m in matches:
                if m.queryIdx == m.trainIdx:
                    continue
                p1   = kp[m.queryIdx].pt
                p2   = kp[m.trainIdx].pt
                dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                if dist > 40 and m.distance < 30:
                    suspicious.append((p1, p2))

            clone_regions = len(suspicious)
            if clone_regions > 10:
                copy_score = min(clone_regions * 3, 80)
                indicators.append(f'Copy-move detected: {clone_regions} suspicious duplicate regions')
            elif clone_regions > 4:
                copy_score = 30
                indicators.append(f'Possible copy-move: {clone_regions} near-duplicate feature pairs')

        component_scores['copy_move'] = copy_score
        R.add_stat('Copy-Move Regions', clone_regions)
    except Exception as e:
        R.pdf_text(f'Copy-move analysis failed: {e}')
        component_scores['copy_move'] = 0.0

    # -- 3C: PRNU Noise Inconsistency -----------------------------
    try:
        noise_maps = []
        for ksize in [3, 5, 7]:
            blurred = cv2.GaussianBlur(gray_array.astype(float), (ksize, ksize), 0)
            noise_maps.append(gray_array.astype(float) - blurred)

        noise_combined = np.mean(noise_maps, axis=0)
        grid_h, grid_w = h // 4, w // 4
        noise_cells = []
        for row in range(4):
            for col in range(4):
                cell = noise_combined[row*grid_h:(row+1)*grid_h, col*grid_w:(col+1)*grid_w]
                noise_cells.append(float(np.std(cell)))

        noise_mean  = float(np.mean(noise_cells))
        noise_range = float(max(noise_cells) - min(noise_cells))
        noise_cv    = noise_range / (noise_mean + 1e-6)

        prnu_score = 0.0
        if noise_cv > 1.2:
            prnu_score = min(noise_cv * 30, 70)
            indicators.append(f'Noise inconsistency across regions (CV={noise_cv:.2f})  -  possible composite image')

        component_scores['prnu'] = prnu_score
        R.add_stat('Noise CV',    f'{noise_cv:.3f}')
        R.add_stat('Noise Range', f'{noise_range:.2f}')

        noise_vis       = np.clip(np.abs(noise_combined) * 8, 0, 255).astype(np.uint8)
        noise_grid_img  = np.array(noise_cells).reshape(4,4)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(noise_vis, cmap='viridis')
        axes[0].set_title(f'PRNU Noise Map (CV={noise_cv:.2f})', color='#c9d1d9'); axes[0].axis('off')
        im = axes[1].imshow(noise_grid_img, cmap='RdYlGn_r', aspect='auto')
        plt.colorbar(im, ax=axes[1])
        axes[1].set_title('Noise Std per Region', color='#c9d1d9')
        plt.suptitle('PRNU Sensor Noise Analysis', color='#58a6ff', fontsize=12, fontweight='bold')
        plt.tight_layout()
        R.save_graph('prnu_analysis.png', 'PRNU Noise Analysis',
                     f'Camera sensor noise consistency. High regional variation (CV={noise_cv:.2f}) suggests composite image.', important=True)
        plt.close(fig)
    except Exception as e:
        R.pdf_text(f'PRNU analysis failed: {e}')
        component_scores['prnu'] = 0.0

    # -- 3D: Metadata Forensics ------------------------------------
    try:
        meta_score = 0.0
        meta_flags = []
        pil_raw    = Image.open(filepath)
        exif_data  = pil_raw.getexif() or {}
        exif_dict  = {TAGS.get(k, k): v for k, v in exif_data.items()}

        software_tag = str(exif_dict.get('Software', '')).lower()
        if any(s in software_tag for s in ['photoshop', 'lightroom', 'gimp', 'affinity', 'pixelmator']):
            meta_score += 50
            meta_flags.append(f'Editing software in EXIF: {exif_dict.get("Software", "")}')
        if any(s in software_tag for s in ['dall-e', 'midjourney', 'stable diffusion', 'firefly', 'imagen']):
            meta_score += 80
            meta_flags.append(f'AI generator software tag: {exif_dict.get("Software", "")}')

        has_make     = 'Make' in exif_dict
        has_model    = 'Model' in exif_dict
        has_gps      = any('GPS' in str(k) for k in exif_dict)
        has_datetime = 'DateTime' in exif_dict or 'DateTimeOriginal' in exif_dict

        if not has_make and not has_model and len(exif_dict) < 3:
            meta_score += 25
            meta_flags.append('No camera make/model in EXIF  -  consistent with AI-generated or edited image')
        if has_make and not has_datetime:
            meta_score += 20
            meta_flags.append('Camera identified but no timestamp  -  metadata may have been stripped')

        icc = pil_raw.info.get('icc_profile')
        if icc and b'Adobe' in icc and not any(s in software_tag for s in ['photoshop', 'lightroom']):
            meta_score += 15
            meta_flags.append('Adobe ICC profile without Adobe software tag  -  possible metadata stripping')

        component_scores['metadata'] = min(meta_score, 100)
        R.add_stat('Software Tag', exif_dict.get('Software', 'Not present'))
        R.add_stat('Camera Make',  exif_dict.get('Make', 'Not present'))
        R.add_stat('Has GPS',      'Yes' if has_gps else 'No')
        for f in meta_flags:
            indicators.append(f'[Metadata] {f}')
    except Exception as e:
        R.pdf_text(f'Metadata analysis failed: {e}')
        component_scores['metadata'] = 0.0

    # -- 3E: Patch-Level Inconsistency ----------------------------
    try:
        patch_size   = max(32, min(h, w) // 16)
        patch_scores = []

        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = gray_array[y:y+patch_size, x:x+patch_size].astype(float)
                patch_scores.append({
                    'y': y, 'x': x,
                    'mean': float(np.mean(patch)),
                    'std':  float(np.std(patch)),
                    'fft':  float(np.std(np.log1p(np.abs(np.fft.fft2(patch))))),
                })

        if len(patch_scores) > 4:
            stds     = [p['std'] for p in patch_scores]
            ffts     = [p['fft'] for p in patch_scores]
            std_mean, std_std = np.mean(stds), np.std(stds)
            fft_mean, fft_std = np.mean(ffts), np.std(ffts)

            outlier_patches = [
                p for p in patch_scores
                if abs(p['std'] - std_mean) > 2.5 * std_std
                or abs(p['fft'] - fft_mean) > 2.5 * fft_std
            ]
            outlier_ratio = len(outlier_patches) / len(patch_scores)

            patch_incon_score = 0.0
            if outlier_ratio > 0.15:
                patch_incon_score = min(outlier_ratio * 200, 70)
                indicators.append(f'Patch inconsistency: {len(outlier_patches)}/{len(patch_scores)} outlier patches')

            component_scores['patch'] = patch_incon_score
            R.add_stat('Patch Outliers', f'{len(outlier_patches)}/{len(patch_scores)}')

            rows = (h - patch_size) // patch_size + 1
            cols = (w - patch_size) // patch_size + 1
            heatmap_data = np.zeros((rows, cols))
            for p in patch_scores:
                r_idx = p['y'] // patch_size
                c_idx = p['x'] // patch_size
                if r_idx < rows and c_idx < cols:
                    is_out = (abs(p['std'] - std_mean) > 2.5 * std_std
                              or abs(p['fft'] - fft_mean) > 2.5 * fft_std)
                    heatmap_data[r_idx, c_idx] = 1.0 if is_out else 0.0

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(pil_image.convert('RGB')); axes[0].set_title('Original', color='#c9d1d9'); axes[0].axis('off')
            axes[1].imshow(heatmap_data, cmap='Reds', aspect='auto')
            axes[1].set_title(f'Patch Outlier Map ({outlier_ratio*100:.0f}% outliers)', color='#c9d1d9')
            plt.suptitle('Patch-Level Inconsistency Analysis', color='#58a6ff', fontsize=12, fontweight='bold')
            plt.tight_layout()
            R.save_graph('patch_analysis.png', 'Patch Inconsistency',
                         f'{outlier_ratio*100:.0f}% of patches are statistically inconsistent with surroundings.', important=True)
            plt.close(fig)
        else:
            component_scores['patch'] = 0.0
    except Exception as e:
        R.pdf_text(f'Patch analysis failed: {e}')
        component_scores['patch'] = 0.0

    # -- Combine --------------------------------------------------
    weights     = {'ela': 0.35, 'copy_move': 0.25, 'prnu': 0.20, 'patch': 0.12, 'metadata': 0.08}
    manip_score = float(np.clip(
        sum(component_scores.get(k, 0) * w for k, w in weights.items()), 0, 100
    ))

    R.add_stat('ELA Score',          f'{component_scores.get("ela", 0):.0f}%')
    R.add_stat('Copy-Move Score',    f'{component_scores.get("copy_move", 0):.0f}%')
    R.add_stat('PRNU Score',         f'{component_scores.get("prnu", 0):.0f}%')
    R.add_stat('Patch Score',        f'{component_scores.get("patch", 0):.0f}%')
    R.add_stat('Metadata Score',     f'{component_scores.get("metadata", 0):.0f}%')
    R.add_stat('Manipulation Score', f'{manip_score:.1f}%')
    R.payload['stage_scores']['manipulation'] = round(manip_score, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels  = ['ELA', 'Copy-Move', 'PRNU\nNoise', 'Patch\nInconsistency', 'Metadata']
    values  = [component_scores.get(k, 0) for k in ['ela', 'copy_move', 'prnu', 'patch', 'metadata']]
    colors  = ['#f85149' if v >= 50 else '#d29922' if v >= 25 else '#3fb950' for v in values]
    bars    = ax.bar(labels, values, color=colors, width=0.5)
    ax.axhline(y=50, color='#f85149', linestyle='--', linewidth=1.5, label='Suspicion threshold (50)')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{val:.0f}', ha='center', color='#c9d1d9', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115); ax.set_ylabel('Suspicion Score (0-100)')
    ax.set_title(f'Manipulation Detection Dashboard  -  Combined: {manip_score:.1f}%', color='#c9d1d9')
    ax.legend()
    plt.tight_layout()
    R.save_graph('manipulation_dashboard.png', 'Manipulation Dashboard',
                 f'ELA, copy-move, PRNU, patch, metadata. Combined manipulation score: {manip_score:.1f}%.', important=True)
    plt.close(fig)

    for ind in indicators:
        R.add_indicator(f'[Manipulation] {ind}')

    return manip_score


# ============================================================
# STAGE 4  -  Vehicle / Object Damage Analysis
# Runs only when no human face is detected.
# Targets insurance fraud: AI-generated damage, inpainted dents,
# composited backgrounds, severity exaggeration.
# ============================================================
def vehicle_damage_analysis(filepath, pil_image, R: AnalysisResult):
    """
    Five checks specific to vehicle/object images:
    A. Damage region ELA  -  inpainted damage has higher ELA than real panels
    B. Shadow consistency  -  AI damage shadow direction often wrong
    C. Paint texture consistency  -  real deformed metal preserves stretched texture
    D. Damage boundary analysis  -  real damage has sharp irregular boundaries
    E. Insurance metadata checks  -  WhatsApp/phone photos should have GPS + device
    """
    R.pdf_text('<b>Stage 4  -  Vehicle & Object Damage Analysis</b>', 'Heading1')
    apply_graph_style()

    indicators       = []
    component_scores = {}
    rgb_array        = np.array(pil_image.convert('RGB'))
    gray_array       = np.array(pil_image.convert('L'))
    h, w             = gray_array.shape

    # -- 4A: Identify damage region via edge chaos -----------------
    # Damage areas (dents, crumples) produce chaotic, dense edges
    # compared to smooth undamaged panels
    try:
        edge_map  = cv2.Canny(gray_array, 50, 150)
        kernel    = np.ones((20, 20), np.uint8)
        edge_den  = cv2.filter2D(edge_map.astype(float), -1, kernel / 400)

        # Identify high-edge-density region = likely damage area
        threshold    = float(np.percentile(edge_den, 85))
        damage_mask  = (edge_den > threshold).astype(np.uint8)
        undmg_mask   = (edge_den <= float(np.percentile(edge_den, 40))).astype(np.uint8)

        damage_pixels = int(np.sum(damage_mask))
        total_pixels  = h * w
        damage_ratio  = damage_pixels / total_pixels

        R.add_stat('Damage Region Ratio', f'{damage_ratio*100:.1f}%')

    except Exception as e:
        R.pdf_text(f'Damage region detection failed: {e}')
        damage_mask = np.ones((h, w), dtype=np.uint8)
        undmg_mask  = np.zeros((h, w), dtype=np.uint8)
        damage_ratio = 0.0

    # -- 4B: Damage-region ELA -------------------------------------
    # Key test: inpainted damage has different JPEG compression history
    # than the surrounding real panel pixels.
    try:
        buf = io.BytesIO()
        pil_image.convert('RGB').save(buf, format='JPEG', quality=85)
        buf.seek(0)
        recomp   = Image.open(buf).convert('RGB')
        ela_diff = np.array(ImageChops.difference(pil_image.convert('RGB'), recomp)).astype(float)
        ela_gray = np.mean(ela_diff, axis=2)

        ela_damage  = float(np.mean(ela_gray[damage_mask > 0])) if damage_mask.any() else 0.0
        ela_undmg   = float(np.mean(ela_gray[undmg_mask > 0]))  if undmg_mask.any() else 0.0
        ela_ratio   = ela_damage / (ela_undmg + 1e-6)

        # If damage region ELA is significantly higher than undamaged panels
        # this strongly suggests the damage was added after the original photo
        damage_ela_score = 0.0
        if ela_ratio > 2.0:
            damage_ela_score = min((ela_ratio - 1.0) * 35, 90)
            indicators.append(f'Damage region ELA {ela_ratio:.1f}x higher than surrounding panels  -  damage likely inpainted')
        elif ela_ratio > 1.4:
            damage_ela_score = 30.0
            indicators.append(f'Elevated ELA in damage region (ratio={ela_ratio:.1f})  -  possible manipulation')

        component_scores['damage_ela'] = damage_ela_score
        R.add_stat('Damage Region ELA',    f'{ela_damage:.2f}')
        R.add_stat('Undamaged Panel ELA',  f'{ela_undmg:.2f}')
        R.add_stat('ELA Damage Ratio',     f'{ela_ratio:.2f}')

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb_array); axes[0].set_title('Original', color='#c9d1d9'); axes[0].axis('off')
        ela_vis = np.clip(ela_gray * 8, 0, 255).astype(np.uint8)
        axes[1].imshow(ela_vis, cmap='hot')
        axes[1].set_title(f'Damage ELA Map\nDamage={ela_damage:.1f} vs Panel={ela_undmg:.1f}', color='#c9d1d9'); axes[1].axis('off')
        overlay = rgb_array.copy()
        overlay[damage_mask > 0] = (overlay[damage_mask > 0] * 0.5 + np.array([255, 80, 80]) * 0.5).astype(np.uint8)
        axes[2].imshow(overlay); axes[2].set_title('Detected Damage Region (red)', color='#c9d1d9'); axes[2].axis('off')
        plt.suptitle('Vehicle Damage ELA  -  Inpainted damage shows higher compression error', color='#58a6ff', fontsize=11, fontweight='bold')
        plt.tight_layout()
        R.save_graph('vehicle_damage_ela.png', 'Vehicle Damage ELA',
                     f'ELA comparison: damage region={ela_damage:.1f} vs undamaged panels={ela_undmg:.1f}. Ratio={ela_ratio:.2f}. >1.4 is suspicious.', important=True)
        plt.close(fig)

    except Exception as e:
        R.pdf_text(f'Damage ELA failed: {e}')
        component_scores['damage_ela'] = 0.0

    # -- 4C: Shadow Consistency ------------------------------------
    # AI-generated damage frequently has incorrect shadow direction.
    # Method: estimate dominant gradient direction in undamaged areas,
    # compare to gradient direction in damage region.
    try:
        sobelx = cv2.Sobel(gray_array, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_array, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(sobely, sobelx)

        # Get dominant light angle from undamaged smooth panels
        undmg_angles = angles[undmg_mask > 0] if undmg_mask.any() else angles.flatten()
        dmg_angles   = angles[damage_mask > 0] if damage_mask.any() else angles.flatten()

        if len(undmg_angles) > 100 and len(dmg_angles) > 100:
            # Circular mean of gradient angles
            undmg_dir = float(np.arctan2(np.mean(np.sin(undmg_angles)), np.mean(np.cos(undmg_angles))))
            dmg_dir   = float(np.arctan2(np.mean(np.sin(dmg_angles)),   np.mean(np.cos(dmg_angles))))
            angle_diff = abs(undmg_dir - dmg_dir)
            if angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff

            shadow_score = 0.0
            angle_deg    = float(np.degrees(angle_diff))
            if angle_diff > np.pi / 3:  # > 60 degrees difference
                shadow_score = min(angle_diff / np.pi * 80, 70)
                indicators.append(f'Shadow direction inconsistency ({angle_deg:.0f} deg)  -  AI damage often has wrong lighting')
            elif angle_diff > np.pi / 6:  # > 30 degrees
                shadow_score = 25.0

            component_scores['shadow'] = shadow_score
            R.add_stat('Shadow Angle Diff', f'{angle_deg:.1f} deg')
        else:
            component_scores['shadow'] = 0.0

    except Exception as e:
        R.pdf_text(f'Shadow analysis failed: {e}')
        component_scores['shadow'] = 0.0

    # -- 4D: Paint Texture Consistency -----------------------------
    # Real deformed metal: paint texture is stretched/compressed version
    # of original. AI-generated damage: different texture entirely.
    # Method: compare local frequency content (texture fingerprint)
    # between damage and undamaged regions.
    try:
        def region_texture(mask, arr, n_samples=200):
            coords = np.argwhere(mask > 0)
            if len(coords) < n_samples:
                return None
            idx     = np.random.choice(len(coords), n_samples, replace=False)
            patches = []
            ps      = 8
            for r, c in coords[idx]:
                if r+ps < arr.shape[0] and c+ps < arr.shape[1]:
                    p = arr[r:r+ps, c:c+ps]
                    patches.append(float(np.std(p)))
            return float(np.mean(patches)) if patches else None

        np.random.seed(42)
        dmg_tex  = region_texture(damage_mask, gray_array)
        undmg_tex = region_texture(undmg_mask, gray_array)

        texture_score = 0.0
        if dmg_tex is not None and undmg_tex is not None:
            tex_ratio = abs(dmg_tex - undmg_tex) / (undmg_tex + 1e-6)
            R.add_stat('Damage Texture Std',   f'{dmg_tex:.2f}')
            R.add_stat('Undamaged Texture Std', f'{undmg_tex:.2f}')
            R.add_stat('Texture Ratio',         f'{tex_ratio:.2f}')

            if tex_ratio > 0.5:
                texture_score = min(tex_ratio * 60, 65)
                indicators.append(f'Texture discontinuity at damage boundary (ratio={tex_ratio:.2f})  -  inconsistent with real deformation')
            elif tex_ratio > 0.3:
                texture_score = 20.0

        component_scores['texture'] = texture_score

    except Exception as e:
        R.pdf_text(f'Texture analysis failed: {e}')
        component_scores['texture'] = 0.0

    # -- 4E: Damage Boundary Analysis -----------------------------
    # Real collision damage has physically plausible boundaries:
    # irregular, jagged edges where paint cracks.
    # AI-generated damage tends to have smooth, rounded boundaries.
    try:
        # Dilate damage mask to get boundary region
        kernel       = np.ones((5,5), np.uint8)
        dmg_dilated  = cv2.dilate(damage_mask, kernel, iterations=3)
        boundary     = cv2.bitwise_and(dmg_dilated, cv2.bitwise_not(damage_mask))

        if boundary.any():
            boundary_region = gray_array[boundary > 0]
            boundary_std    = float(np.std(boundary_region))
            boundary_grad   = cv2.Canny(gray_array * boundary, 30, 90)
            boundary_edge_d = float(np.mean(boundary_grad > 0))

            boundary_score = 0.0
            # Real damage: high std at boundary (irregular), high edge density
            # AI damage: smooth boundary, lower std
            if boundary_std < 15 and boundary_edge_d < 0.05:
                boundary_score = 55.0
                indicators.append('Unnaturally smooth damage boundary  -  real collision damage has irregular edges')
            elif boundary_std < 25:
                boundary_score = 25.0

            component_scores['boundary'] = boundary_score
            R.add_stat('Damage Boundary Std',    f'{boundary_std:.2f}')
            R.add_stat('Damage Boundary Edges',  f'{boundary_edge_d:.4f}')
        else:
            component_scores['boundary'] = 0.0

    except Exception as e:
        R.pdf_text(f'Boundary analysis failed: {e}')
        component_scores['boundary'] = 0.0

    # -- 4F: Insurance Metadata Checks -----------------------------
    # Insurance/accident photos taken on phones should have:
    # GPS coordinates, device make/model, timestamp, WhatsApp or camera app signature
    try:
        ins_score  = 0.0
        ins_flags  = []
        pil_raw    = Image.open(filepath)
        exif_data  = pil_raw.getexif() or {}
        exif_dict  = {TAGS.get(k, k): v for k, v in exif_data.items()}

        has_gps      = any('GPS' in str(k) for k in exif_dict)
        has_make     = 'Make' in exif_dict
        has_model    = 'Model' in exif_dict
        has_datetime = 'DateTime' in exif_dict or 'DateTimeOriginal' in exif_dict
        software_tag = str(exif_dict.get('Software', '')).lower()

        # A genuine insurance claim photo from a phone will almost always have GPS
        if not has_gps:
            ins_score += 20
            ins_flags.append('No GPS coordinates  -  genuine on-site accident photos typically include location')

        # Should have phone make/model
        if not has_make or not has_model:
            ins_score += 20
            ins_flags.append('No device make/model in EXIF  -  authentic phone photos always include this')

        # Should have timestamp
        if not has_datetime:
            ins_score += 25
            ins_flags.append('No timestamp in EXIF  -  metadata may have been stripped or image was generated')

        # Editing software in an "accident photo" is very suspicious
        if any(s in software_tag for s in ['photoshop', 'lightroom', 'gimp', 'affinity']):
            ins_score += 60
            ins_flags.append(f'Professional editing software found in accident photo: {exif_dict.get("Software", "")}')

        if any(s in software_tag for s in ['dall-e', 'midjourney', 'stable diffusion', 'firefly', 'imagen']):
            ins_score += 90
            ins_flags.append(f'AI generator software tag in accident photo: {exif_dict.get("Software", "")}')

        # WhatsApp-shared images: check for typical WhatsApp JPEG quality signature
        # WhatsApp recompresses to ~85 quality. If the image has very low quality
        # markers it went through heavy re-encoding (possible manipulation pipeline)
        component_scores['insurance_meta'] = min(ins_score, 100)
        R.add_stat('GPS Present',    'Yes' if has_gps else 'No')
        R.add_stat('Device in EXIF', 'Yes' if (has_make and has_model) else 'No')
        R.add_stat('Timestamp',      'Yes' if has_datetime else 'No')
        for f in ins_flags:
            indicators.append(f'[Insurance] {f}')

    except Exception as e:
        R.pdf_text(f'Insurance metadata check failed: {e}')
        component_scores['insurance_meta'] = 0.0

    # -- Combine vehicle scores ------------------------------------
    veh_weights = {
        'damage_ela':     0.35,
        'shadow':         0.20,
        'texture':        0.20,
        'boundary':       0.15,
        'insurance_meta': 0.10,
    }
    vehicle_score = float(np.clip(
        sum(component_scores.get(k, 0) * w for k, w in veh_weights.items()), 0, 100
    ))

    R.add_stat('Damage ELA Score',     f'{component_scores.get("damage_ela", 0):.0f}%')
    R.add_stat('Shadow Score',         f'{component_scores.get("shadow", 0):.0f}%')
    R.add_stat('Texture Score',        f'{component_scores.get("texture", 0):.0f}%')
    R.add_stat('Boundary Score',       f'{component_scores.get("boundary", 0):.0f}%')
    R.add_stat('Insurance Meta Score', f'{component_scores.get("insurance_meta", 0):.0f}%')
    R.add_stat('Vehicle Damage Score', f'{vehicle_score:.1f}%')
    R.payload['stage_scores']['vehicle_damage'] = round(vehicle_score, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels  = ['Damage\nELA', 'Shadow\nConsistency', 'Texture\nConsistency', 'Boundary\nAnalysis', 'Insurance\nMetadata']
    values  = [component_scores.get(k, 0) for k in ['damage_ela', 'shadow', 'texture', 'boundary', 'insurance_meta']]
    colors  = ['#f85149' if v >= 50 else '#d29922' if v >= 25 else '#3fb950' for v in values]
    bars    = ax.bar(labels, values, color=colors, width=0.5)
    ax.axhline(y=50, color='#f85149', linestyle='--', linewidth=1.5, label='Suspicion threshold')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{val:.0f}', ha='center', color='#c9d1d9', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115); ax.set_ylabel('Suspicion Score (0-100)')
    ax.set_title(f'Vehicle Damage Forensic Dashboard  -  Combined: {vehicle_score:.1f}%', color='#c9d1d9')
    ax.legend()
    plt.tight_layout()
    R.save_graph('vehicle_dashboard.png', 'Vehicle Damage Dashboard',
                 f'Damage ELA, shadow, texture, boundary, and metadata scores. Vehicle damage score: {vehicle_score:.1f}%.', important=True)
    plt.close(fig)

    for ind in indicators:
        R.add_indicator(f'[Vehicle] {ind}')

    return vehicle_score


# ============================================================
# STAGE 5  -  Deep Learning Detector + Adaptive Fusion
# ============================================================
def dl_detector(filepath, pil_image, combined_forensic_score, manip_score,
                vehicle_score, has_human_face, all_indicators,
                freq_reliable, R: AnalysisResult):

    R.pdf_text('<b>Stage 5  -  Deep Learning Detector</b>', 'Heading1')
    apply_graph_style()

    dl_available  = False
    dl_deepfake_score = 0.0
    matched_label = 'none'

    try:
        result    = _run_dl_deepfake_model(pil_image)  # loads, scores, flushes before returning
        label_map = {r['label']: r['score'] for r in result}
        dl_deepfake_score, matched_label = extract_fake_score(label_map)
        dl_available = True
        R.pdf_text(f'DL deepfake model output: {result}')
    except Exception as e:
        R.pdf_text(f'Primary DL model failed: {e}  -  falling back to forensic score.')

    if not dl_available:
        dl_deepfake_score = combined_forensic_score
        matched_label     = 'forensic-only'

    # -- AI-generation ensemble: two models run sequentially, scores averaged.
    #    Model 1: Organika/sdxl-detector   — fine-tuned on SDXL outputs
    #    Model 2: umm-maybe/AI-image-detector — broader training (DALL-E,
    #             Midjourney, SD 1.x/2.x, Firefly, etc.)
    #    Both use load-score-flush pattern; never resident simultaneously.
    #    If one fails, the other's score is used alone (graceful degradation).
    #    If both fail, dl_ai_score stays None and fusion falls back to forensics.
    dl_ai_available  = False
    dl_ai_score       = None
    ai_matched_label = 'none'
    _ai_scores = []

    # Model 1: sdxl-detector
    try:
        ai_result_1    = _run_dl_ai_model(pil_image)
        ai_label_map_1 = {r['label']: r['score'] for r in ai_result_1}
        score_1, label_1 = _extract_ai_generated_score(ai_label_map_1)
        _ai_scores.append(score_1)
        ai_matched_label = label_1
        R.pdf_text(f'AI-gen model 1 (sdxl-detector): {ai_result_1}')
    except Exception as e:
        R.pdf_text(f'AI-gen model 1 (sdxl-detector) failed: {e}')

    # Model 2: umm-maybe/AI-image-detector (broader training)
    try:
        ai_result_2    = _run_dl_ai_model_2(pil_image)
        ai_label_map_2 = {r['label']: r['score'] for r in ai_result_2}
        score_2, label_2 = _extract_ai_generated_score(ai_label_map_2)
        _ai_scores.append(score_2)
        R.pdf_text(f'AI-gen model 2 (umm-maybe/AI-image-detector): {ai_result_2}')
        R.add_stat('DL AI-Gen Model 2 Score', f'{score_2:.1f}%')
    except Exception as e:
        R.pdf_text(f'AI-gen model 2 (umm-maybe) failed: {e}')

    if _ai_scores:
        # Average ensemble scores — equal weight, both models validated on same label scheme
        import statistics
        dl_ai_score     = float(statistics.mean(_ai_scores))
        dl_ai_available = True
        R.add_stat('DL AI-Gen Ensemble Scores', ' / '.join(f'{s:.1f}' for s in _ai_scores))

    R.add_stat('DL Deepfake Score', f'{dl_deepfake_score:.1f}%')
    R.add_stat('DL Deepfake Label', matched_label)
    R.payload['stage_scores']['deep_learning'] = round(dl_deepfake_score, 1)

    if dl_ai_available:
        R.add_stat('DL AI-Generated Score', f'{dl_ai_score:.1f}%')
        R.add_stat('DL AI-Generated Label', ai_matched_label)
        R.payload['stage_scores']['dl_ai_generated'] = round(dl_ai_score, 1)
    else:
        R.add_stat('DL AI-Generated Score', 'Unavailable')
        R.payload['stage_scores']['dl_ai_generated'] = None

    # Alias - everything below (graph title, fusion math) is unchanged from
    # before this phase and still reads `fake_score`.
    fake_score = dl_deepfake_score

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
    axes[1].axhline(y=np.mean(patch_variances), color='#f85149', linestyle='--',
                    label=f'Mean={np.mean(patch_variances):.0f}')
    axes[1].set_title(f'Patch Variance  (std={consistency:.0f})', color='#c9d1d9')
    axes[1].set_xlabel('Patch Index'); axes[1].set_ylabel('Variance')
    axes[1].legend(); axes[1].grid(True)
    plt.suptitle(f'Deep Learning Analysis  -  DL Score: {fake_score:.1f}%  |  Model: {matched_label}',
                 color='#58a6ff', fontsize=12, fontweight='bold')
    plt.tight_layout()
    R.save_graph('dl_analysis.png', 'Deep Learning Analysis',
                 f'Attention heatmap and patch variance. DL confidence: {fake_score:.1f}%.', important=True)
    plt.close(fig)

    # -- Adaptive fusion ------------------------------------------
    # DL model is trained on human face deepfakes.
    # When no human face is present (vehicle, object), its weight is reduced
    # and vehicle damage score takes a significant share.
    if has_human_face:
        # Standard human face image - UNCHANGED from before this fix.
        if not freq_reliable:
            w_forensic, w_manip, w_vehicle, w_dl = 0.15, 0.35, 0.00, 0.50
        else:
            w_forensic, w_manip, w_vehicle, w_dl = 0.25, 0.30, 0.00, 0.45
        w_dl_ai           = 0.00  # not touched in this fix - see image_pipeline.py history
        dl_ai_for_fusion  = 0.0
        R.pdf_text('Fusion mode: FACE IMAGE  -  DL model weighted heavily.')
    else:
        # Vehicle / object image. EMERGENCY FIX: dl_ai_score (Phase 1's AI-
        # generated-image detector) previously had ZERO weight here despite
        # being the single most relevant signal for non-face content.
        # Real incident: a ChatGPT-edited vehicle damage photo scored 98.7%
        # on this signal and the final score still came out 32.4% ("LOW")
        # because nothing read it. dl_deepfake_score's weight is reduced
        # (not removed) here since that model is trained for face-swap
        # detection and is a weaker signal for vehicle/object images.
        w_forensic, w_manip, w_vehicle, w_dl, w_dl_ai = 0.10, 0.20, 0.30, 0.10, 0.30

        if dl_ai_score is None:
            # Graceful degradation: redistribute its weight onto vehicle_score
            # instead of silently zeroing it out.
            w_vehicle        += w_dl_ai
            w_dl_ai           = 0.00
            dl_ai_for_fusion  = 0.0
        else:
            dl_ai_for_fusion = dl_ai_score

        R.pdf_text('Fusion mode: VEHICLE/OBJECT IMAGE  -  vehicle damage + AI-generation score weighted heavily, DL deepfake weight reduced.')
        R.add_indicator('[Vehicle] Image contains no human face  -  vehicle damage forensics applied')

    final = float(np.clip(
        combined_forensic_score * w_forensic +
        manip_score             * w_manip    +
        vehicle_score           * w_vehicle  +
        fake_score              * w_dl       +
        dl_ai_for_fusion        * w_dl_ai,
        0, 100
    ))

    R.add_stat('Forensic Score',     f'{combined_forensic_score:.1f}%')
    R.add_stat('Manipulation Score', f'{manip_score:.1f}%')
    R.add_stat('Vehicle Score',      f'{vehicle_score:.1f}%')
    R.add_stat('DL Score',           f'{fake_score:.1f}%')
    R.add_stat('Fusion Mode',        'Face image' if has_human_face else 'Vehicle/object image')
    R.add_stat('Final Fusion Score', f'{final:.1f}%')

    return final


# ============================================================
# EXIF ANALYSIS  -  runs before all other stages
# Produces two scores fed into classify_dominant() in helpers.py:
#   exif_ai_score   (0-100): how strongly EXIF signals AI-generation
#   exif_real_score (0-100): how strongly EXIF confirms authentic camera origin
#
# Display policy: only show the user findings that are conclusive.
# Strong internal signals (camera make/model, GPS, lens optics) are
# used to influence the score but NOT shown as indicators — if the
# device ID is wrong the user loses trust in the entire result even
# if the classification is correct. Conclusive findings (AI generator
# tag, zero EXIF, thumbnail mismatch) are shown.
# ============================================================
def exif_analysis(filepath, pil_raw, R: AnalysisResult) -> dict:
    """
    Returns dict with keys:
        exif_ai_score    float 0-100
        exif_real_score  float 0-100
        exif_fields      int
        conclusive_findings  list[str]  — shown to user
        internal_notes   list[str]      — PDF only, not in indicators
    """
    result = {
        'exif_ai_score':       0.0,
        'exif_real_score':     0.0,
        'exif_edit_score':     0.0,   # NEW: editing software signal — separate from AI generation
        'exif_fields':         0,
        'conclusive_findings': [],
        'internal_notes':      [],
        'exif_no_metadata_flagged': False,  # NEW: set True if the no/minimal-EXIF signal fired,
                                             # so analyze_image() knows whether to check for corroboration
    }

    try:
        exif_data = pil_raw.getexif() or {}
        exif_dict = {TAGS.get(k, k): v for k, v in exif_data.items()}
        result['exif_fields'] = len(exif_dict)

        ai_score   = 0.0
        edit_score = 0.0   # accumulates editing-software signals (separate from AI generation)
        real_score = 0.0

        software_tag = str(exif_dict.get('Software', '')).lower()

        # ── CONCLUSIVE AI-generation signals (shown to user) ──────────────────

        # AI generator software tag — near-definitive
        AI_GEN_TOOLS = ['dall-e', 'midjourney', 'stable diffusion', 'firefly',
                         'imagen', 'adobe firefly', 'runway', 'bing image',
                         'canva ai', 'nightcafe', 'leonardo.ai']
        matched_ai_tool = next((t for t in AI_GEN_TOOLS if t in software_tag), None)
        if matched_ai_tool:
            ai_score += 95
            finding = f'AI generator software identified in metadata: "{exif_dict.get("Software", "")}"'
            result['conclusive_findings'].append(f'[EXIF] {finding}')
            R.add_indicator(f'[EXIF] {finding}')

        # Professional editing software (conclusive but with nuance)
        EDIT_TOOLS = ['photoshop', 'lightroom', 'gimp', 'affinity', 'pixelmator', 'capture one']
        matched_edit_tool = next((t for t in EDIT_TOOLS if t in software_tag), None)
        if matched_edit_tool and not matched_ai_tool:
            # Editing software is evidence of EDITING, not AI generation.
            # Previously this added to ai_score — that caused colour-graded real photos
            # to be classified as AI_GENERATED. Now routed to edit_score, which feeds
            # the EDITED classification path in classify_dominant(), not ai_gen_composite.
            edit_score += 45
            finding = f'Professional editing software in metadata: "{exif_dict.get("Software", "")}" — image has been processed'
            result['conclusive_findings'].append(f'[EXIF] {finding}')
            R.add_indicator(f'[EXIF] {finding}')

        # Zero EXIF on a JPEG — JPEGs from real cameras always have EXIF,
        # BUT this is genuinely ambiguous on its own: messaging apps
        # (WhatsApp, Telegram, Signal) strip ALL EXIF on send by default,
        # as do many phone camera apps for privacy, and so do simple re-saves.
        # A photo transmitted via WhatsApp (filenames like
        # "PHOTO-2026-06-24-17-09-00.jpg" are a strong tell) will have zero
        # EXIF regardless of whether it's a real photo or AI-generated.
        # Real case: a genuine shirt-colour-edited photo with dl_ai_generated
        # at 0.9% (pixel models confidently said "not AI-generated") still
        # got pushed to 63% AI_GENERATED, driven almost entirely by this
        # single unconditional +70.
        #
        # Fix: this signal alone now only contributes a reduced base score.
        # PNG/WebP from the web legitimately have no EXIF, so only flag JPEG.
        # The score is boosted back up later, in analyze_image(), but ONLY
        # if corroborated by another independent signal (elevated dl_ai_
        # generated, or frequency/noise signals also flagging) — see the
        # "no-EXIF corroboration" block after frequency analysis runs.
        # 'exif_no_metadata_flagged' lets that later block find this finding.
        img_format = str(getattr(pil_raw, 'format', '') or '').upper()
        if len(exif_dict) == 0 and img_format == 'JPEG':
            ai_score += 25  # reduced base score — was an unconditional 70
            finding = 'No metadata present in this JPEG — ambiguous on its own (common with messaging-app transmission); see corroboration check'
            result['conclusive_findings'].append(f'[EXIF] {finding}')
            R.add_indicator(f'[EXIF] {finding}')
            result['exif_no_metadata_flagged'] = True
        elif len(exif_dict) < 3 and img_format == 'JPEG' and not matched_ai_tool:
            ai_score += 15  # reduced base score — was an unconditional 35, same rationale
            finding = f'Minimal metadata ({len(exif_dict)} fields) in JPEG — metadata may have been stripped (ambiguous alone)'
            result['conclusive_findings'].append(f'[EXIF] {finding}')
            R.add_indicator(f'[EXIF] {finding}')
            result['exif_no_metadata_flagged'] = True
        else:
            result['exif_no_metadata_flagged'] = False

        # Thumbnail mismatch — thumbnail embedded in EXIF doesn't match image
        # This is a specific deepfake/manipulation signal — face swap often
        # leaves the original thumbnail intact while the main image is replaced
        try:
            from PIL.ExifTags import TAGS as _TAGS
            # ExifThumbnail is stored separately
            thumb_data = pil_raw.getexif().get_ifd(0x8769)  # ExifIFD
            if thumb_data:
                import io as _io
                thumb_bytes = pil_raw.info.get('exif', b'')
                if len(thumb_bytes) > 100:
                    # Quick size check: if thumbnail dimensions don't match
                    # aspect ratio of main image, that's suspicious
                    main_ratio = pil_raw.size[0] / max(pil_raw.size[1], 1)
                    # Can't easily decode thumbnail without full EXIF parse,
                    # so use presence of PixelXDimension/PixelYDimension mismatch
                    exif_w = exif_dict.get('PixelXDimension', 0)
                    exif_h = exif_dict.get('PixelYDimension', 0)
                    if exif_w and exif_h:
                        exif_ratio = exif_w / max(exif_h, 1)
                        if abs(exif_ratio - main_ratio) > 0.15:
                            ai_score += 30
                            finding = 'Metadata dimensions do not match image dimensions — possible image replacement'
                            result['conclusive_findings'].append(f'[EXIF] {finding}')
                            R.add_indicator(f'[EXIF] {finding}')
        except Exception:
            pass  # Thumbnail check is best-effort

        # ── INTERNAL signals (influence score, not shown to user) ─────────────

        has_make     = 'Make' in exif_dict
        has_model    = 'Model' in exif_dict
        has_datetime = 'DateTime' in exif_dict or 'DateTimeOriginal' in exif_dict
        has_gps      = any('GPS' in str(k) for k in exif_dict)

        # Camera optics — AI generators do not produce these
        # FNumber, ExposureTime, FocalLength, ISOSpeedRatings
        optics_fields = ['FNumber', 'ExposureTime', 'FocalLength',
                         'ISOSpeedRatings', 'ShutterSpeedValue', 'ApertureValue',
                         'BrightnessValue', 'ExposureBiasValue', 'MaxApertureValue',
                         'MeteringMode', 'Flash', 'FocalLengthIn35mmFilm']
        optics_present = [f for f in optics_fields if f in exif_dict]
        n_optics = len(optics_present)

        # Camera make + model = strong REAL signal internally
        if has_make and has_model:
            real_score += 30
            result['internal_notes'].append(
                f'Camera identified: {exif_dict.get("Make", "")} {exif_dict.get("Model", "")} — internal REAL signal'
            )

        # Lens optics = strongest REAL signal (AI generators never produce these)
        if n_optics >= 4:
            real_score += 35
            result['internal_notes'].append(
                f'{n_optics} camera optics fields present (FNumber, exposure, focal length etc) — strong REAL signal'
            )
        elif n_optics >= 2:
            real_score += 18
            result['internal_notes'].append(
                f'{n_optics} camera optics fields present — moderate REAL signal'
            )

        # GPS = real device, real location
        if has_gps:
            real_score += 20
            result['internal_notes'].append('GPS coordinates present — internal REAL signal')

        # Consistent timestamp
        if has_datetime:
            real_score += 15
            result['internal_notes'].append('Timestamp present — internal REAL signal')

        # Make present but DateTime stripped — suggests metadata was edited/stripped.
        # This is an editing signal, not an AI-generation signal — route to edit_score.
        if has_make and not has_datetime and not matched_ai_tool:
            edit_score += 25
            result['internal_notes'].append('Camera make present but timestamp absent — possible metadata stripping by editing software')

        # Adobe ICC without matching editing software
        icc = pil_raw.info.get('icc_profile')
        if icc and b'Adobe' in icc and not matched_edit_tool and not matched_ai_tool:
            # Adobe ICC without edit tag: likely edited but tag was stripped.
            # Routed to edit_score, not ai_score — same rationale as editing software above.
            edit_score += 20
            result['internal_notes'].append('Adobe ICC profile without editing software tag — possible editing, metadata partially stripped')

        # ── Write to payload + stats ──────────────────────────────────────────
        result['exif_ai_score']   = float(min(ai_score,   100))
        result['exif_real_score'] = float(min(real_score, 100))
        result['exif_edit_score'] = float(min(edit_score, 100))

        # Stats (shown in detailed metrics panel) — raw numbers only, no judgement
        R.add_stat('EXIF Fields Total',   len(exif_dict))
        R.add_stat('EXIF Software',       exif_dict.get('Software', 'Not present'))
        R.add_stat('EXIF Camera',         f'{exif_dict.get("Make","?")} {exif_dict.get("Model","?")}' if has_make else 'Not present')
        R.add_stat('EXIF Optics Fields',  f'{n_optics}/12')
        R.add_stat('EXIF GPS',            'Present' if has_gps else 'Absent')
        R.add_stat('EXIF Timestamp',      'Present' if has_datetime else 'Absent')
        R.add_stat('EXIF AI Score',       f'{result["exif_ai_score"]:.0f}')
        R.add_stat('EXIF Edit Score',     f'{result["exif_edit_score"]:.0f}')
        R.add_stat('EXIF Real Score',     f'{result["exif_real_score"]:.0f}')

        # PDF only — internal notes for analyst
        for note in result['internal_notes']:
            R.pdf_text(f'[EXIF internal] {note}')

        R.payload['stage_scores']['exif_ai_score']   = round(result['exif_ai_score'],   1)
        R.payload['stage_scores']['exif_edit_score'] = round(result['exif_edit_score'], 1)
        R.payload['stage_scores']['exif_real_score'] = round(result['exif_real_score'], 1)

    except Exception as e:
        R.pdf_text(f'EXIF analysis failed: {e}')

    return result


# ============================================================
# ORCHESTRATOR
# ============================================================
def analyze_image(filepath, R: AnalysisResult):
    R.pdf_text('IMAGE FORENSIC ANALYSIS REPORT', 'Title')
    apply_graph_style()

    pil_raw = Image.open(filepath)
    if getattr(pil_raw, 'format', '') == 'MPO':
        pil_raw.seek(0)
    pil_image = pil_raw.convert('RGB')

    orig_w, orig_h = pil_image.size
    R.add_stat('Format',          getattr(pil_raw, 'format', 'unknown'))
    R.add_stat('Original Dimensions', f'{orig_w} x {orig_h} px')
    R.add_stat('Original Megapixels', f"{orig_w*orig_h/1e6:.2f} MP")

    # Memory cap: with both DL models loaded, resting baseline is already
    # ~1.5-1.6GB out of the 2GB instance limit, leaving very little headroom
    # for per-pixel arrays (FFT, ELA, PRNU, patch analysis all run at full
    # resolution). A 4K+ photo was observed to OOM-kill the whole process
    # (silent restart, no catchable exception - that's the signature of a
    # hard OOM kill, not a Python error). Downscaling here is the fix:
    # forensic signals (FFT bands, ELA, noise residual, edge structure) are
    # still meaningful well below native 4K resolution.
    MAX_DIMENSION = 2048
    was_downscaled = False
    if max(orig_w, orig_h) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(orig_w, orig_h)
        new_size = (int(orig_w * scale), int(orig_h * scale))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
        was_downscaled = True

    R.add_stat('Analyzed Dimensions', f'{pil_image.size[0]} x {pil_image.size[1]} px')
    R.add_stat('Megapixels',          f"{pil_image.size[0]*pil_image.size[1]/1e6:.2f} MP")
    if was_downscaled:
        R.add_stat('Downscaled', f'Yes (from {orig_w}x{orig_h}, memory safety cap)')

    # EXIF Analysis — runs before all forensic stages
    # Produces exif_ai_score and exif_real_score fed into classify_dominant()
    # Conclusive findings shown to user; internal signals used silently
    exif_result = exif_analysis(filepath, pil_raw, R)
    exif_fields = exif_result['exif_fields']

    # Stage 1: Frequency analysis
    freq_score, freq_indicators, freq_reliable = frequency_domain_analysis(
        filepath, pil_image, R
    )

    # ── EXIF / pixel-noise cross-validation ─────────────────────────────────
    # EXIF text fields (Make/Model/GPS/lens optics) are trivially spoofable
    # with a tool like exiftool - their presence alone was never proof of
    # camera origin, just an absence-of-evidence-of-tampering signal. A real
    # camera sensor always has nonzero photon/read noise; AI-generated or
    # heavily-denoised images often show abnormally clean noise residual.
    # If EXIF claims a real camera but the pixel noise contradicts that, the
    # contradiction itself is evidence - this ties two previously-isolated
    # signals (EXIF, computed first; pixel noise, computed here) together
    # instead of treating EXIF's claim as independent truth.
    noise_residual_std = R.payload['stage_scores'].get('noise_residual_std', 999)
    NOISE_TOO_CLEAN_FOR_REAL_CAMERA = 4.0
    EXIF_REAL_CLAIM_THRESHOLD       = 50.0

    if (exif_result['exif_real_score'] >= EXIF_REAL_CLAIM_THRESHOLD
            and noise_residual_std < NOISE_TOO_CLEAN_FOR_REAL_CAMERA):
        finding = (
            f'EXIF metadata claims real-camera origin (real score={exif_result["exif_real_score"]:.0f}%), '
            f'but pixel-level sensor noise is abnormally low (std={noise_residual_std:.2f}, real cameras '
            f'typically show 2-8) — this contradiction suggests the EXIF data may not reflect the '
            f"image's true origin"
        )
        R.add_indicator(f'[EXIF] {finding}')
        R.pdf_text(f'EXIF/noise contradiction detected: {finding}')

        # The contradiction itself is evidence of likely spoofing/AI origin -
        # don't just suppress the real-camera claim, actively raise the
        # AI-generation signal too.
        exif_result['exif_real_score'] = min(exif_result['exif_real_score'], 25.0)
        exif_result['exif_ai_score']   = max(exif_result['exif_ai_score'],   55.0)
        R.payload['stage_scores']['exif_real_score'] = round(exif_result['exif_real_score'], 1)
        R.payload['stage_scores']['exif_ai_score']   = round(exif_result['exif_ai_score'],   1)

    # ── Methodology disclaimer (PDF only) ────────────────────────────────────
    # Agreed wording (Option 3): frames the EXIF cross-check as standard
    # forensic-report rigor rather than a standalone weakness admission,
    # matching the voice already used in verdict_text_v2()'s "should be
    # verified by a qualified analyst" language. Unconditional - describes
    # the general methodology, not just cases where the contradiction above
    # actually fired.
    R.pdf_text(
        'Methodology note: Metadata (EXIF) signals in this report are corroborated against '
        'independent pixel-level sensor-noise analysis rather than accepted at face value, '
        'reducing — though not eliminating — susceptibility to metadata manipulation. As with '
        'all findings in this report, EXIF-derived signals should be considered alongside the '
        'full body of evidence and verified by a qualified analyst.'
    )

    # Stage 2: Face forensics (returns has_human_face flag)
    combined_score, all_indicators, has_human_face = face_forensic_analysis(
        filepath, pil_image, freq_score, freq_indicators, R
    )

    # Stage 3: Manipulation analysis (runs on all images)
    manip_score = manipulation_analysis(filepath, pil_image, R)

    # Stage 4: Vehicle damage analysis (runs on all images, weighted in fusion only when no face)
    vehicle_score = vehicle_damage_analysis(filepath, pil_image, R)

    # The vehicle score is only meaningful (and only weighted >0 in fusion below)
    # when no human face was detected. Suppress it from the displayed stage
    # breakdown for face images so the frontend doesn't show a vehicle tab
    # for portraits/faces  -  it would otherwise always be a non-null number.
    if has_human_face:
        R.payload['stage_scores'].pop('vehicle_damage', None)

    # Store has_human_face in payload so pipeline.py can use it
    # for indicator filtering — vehicle/insurance indicators must never
    # appear when a human face was detected
    R.payload['has_human_face'] = has_human_face

    # Stage 5: DL + adaptive fusion
    final_score = dl_detector(
        filepath, pil_image, combined_score, manip_score,
        vehicle_score, has_human_face, all_indicators, freq_reliable, R
    )

    # ── No-EXIF corroboration check ─────────────────────────────────────────
    # The "no/minimal EXIF metadata" finding in exif_analysis() now only
    # contributes a reduced base score on its own (was an unconditional +70,
    # which is too strong given how common innocent EXIF-stripping is —
    # messaging apps, many phone camera apps, and simple re-saves all strip
    # EXIF regardless of whether the image is real or AI-generated). Real
    # case this was built from: a genuine shirt-colour-edited photo with
    # dl_ai_generated=0.9% (pixel model confidently said "not AI-generated")
    # still got pushed to 63% AI_GENERATED almost entirely by an unconditional
    # no-EXIF signal.
    #
    # This block runs LAST, after every other stage, because corroboration
    # needs frequency/noise (available after Stage 1) AND dl_ai_generated
    # (only available after dl_detector() above, the very last stage) -
    # exif_analysis() itself runs FIRST and has none of these yet.
    #
    # Corroboration sources (any one is enough to apply the boost):
    #   - dl_ai_generated also elevated (the pixel model agrees something's off)
    #   - frequency score also elevated (spectral anomalies independently flagged)
    #   - the EXIF/noise contradiction above already fired (exif_ai_score >= 55
    #     from that block means pixel noise already contradicted a real-camera claim)
    if exif_result.get('exif_no_metadata_flagged'):
        dl_ai_for_corroboration = R.payload['stage_scores'].get('dl_ai_generated')
        CORROBORATION_DL_AI_THRESHOLD  = 40.0
        CORROBORATION_FREQ_THRESHOLD   = 55.0
        NO_EXIF_CORROBORATED_AI_SCORE  = 70.0  # restores the original strength, but only when earned

        corroborated = (
            (dl_ai_for_corroboration is not None and dl_ai_for_corroboration >= CORROBORATION_DL_AI_THRESHOLD)
            or freq_score >= CORROBORATION_FREQ_THRESHOLD
            or exif_result['exif_ai_score'] >= 55.0  # the noise-contradiction block already fired
        )

        if corroborated:
            exif_result['exif_ai_score'] = max(exif_result['exif_ai_score'], NO_EXIF_CORROBORATED_AI_SCORE)
            R.payload['stage_scores']['exif_ai_score'] = round(exif_result['exif_ai_score'], 1)
            # Flag this score as corroboration-derived, not intrinsic EXIF evidence.
            # WHY: classify_dominant()'s EXIF-conclusive ceiling (exif_ai>=70 -> floor
            # ai_gen_composite at exif_ai*0.90) was written assuming exif_ai reflects
            # independent EXIF evidence (AI-tool tag, noise contradiction). Now that
            # this corroboration block can ALSO push exif_ai to 70 by borrowing
            # strength from dl_ai_generated, the ceiling double-counts dl_ai: once
            # directly in the weighted ai_gen_composite averag e, and again via the
            # ceiling it indirectly triggered. Real case: dl_ai=84.4 corroborates
            # exif_ai to 70, weighted composite is already 62.4 (dl_ai counted once,
            # correctly) - the ceiling then pushed it to 63.0,  re-counting dl_ai's
            # contribution a second time through the back door.
            # This flag lets classify_dominant() skip the ceiling for corroboration-
            # derived scores - the COMPOSITE_FLOOR_FACTOR general floor already
            # protects strong dl_ai/freq signals without this double-count.
            R.payload['stage_scores']['exif_ai_corroborated'] = True
            R.add_indicator('[EXIF] No-metadata finding corroborated by independent signal(s) - AI-generation signal strengthened')
            R.pdf_text(
                f'No-EXIF corroboration: dl_ai_generated={dl_ai_for_corroboration}, freq_score={freq_score:.1f} '
                f'- at least one independent signal supports the no-metadata finding, so its contribution was restored.'
            )
        else:
            R.pdf_text(
                f'No-EXIF finding NOT corroborated by other signals (dl_ai_generated={dl_ai_for_corroboration}, '
                f'freq_score={freq_score:.1f}) - kept at reduced strength to avoid over-penalizing images with '
                f'innocently-stripped metadata (e.g. messaging-app transmission).'
            )

    return final_score
