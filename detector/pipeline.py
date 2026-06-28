"""
pipeline.py — v5.2
Main pipeline dispatcher.

New in v5.2:
  - For IMAGE files: calls classify_dominant() after analyze_image()
    to produce dominant classification (x% REAL / x% AI_GENERATED / x% DEEPFAKE)
  - Filters indicators to only show those supporting the dominant class
  - All new fields added alongside legacy fields (no breaking changes)
  - Video, audio, document pipelines: unchange d behaviour
"""
import os
import traceback
from datetime import datetime

from .config import IMAGE_FORMATS, VIDEO_FORMATS, AUDIO_FORMATS, DOCUMENT_FORMATS, REPORT_FOLDER
from .result import AnalysisResult
from .helpers import (
    file_metadata, threat_from_score, verdict_text,
    classify_dominant, filter_indicators,
)
from .image_pipeline    import analyze_image
from .video_pipeline    import analyze_video
from .audio_pipeline    import analyze_audio
from .document_pipeline import analyze_document


def run_pipeline(filepath: str, job_id: str) -> dict:
    """Main entry point. Returns the JSON payload dict."""

    R   = AnalysisResult(job_id)
    ext = os.path.splitext(filepath)[1].lower()

    if   ext in IMAGE_FORMATS    : file_type = 'IMAGE'
    elif ext in VIDEO_FORMATS    : file_type = 'VIDEO'
    elif ext in AUDIO_FORMATS    : file_type = 'AUDIO'
    elif ext in DOCUMENT_FORMATS : file_type = 'DOCUMENT'
    else                         : file_type = 'UNSUPPORTED'

    R.payload['file_type'] = file_type
    R.payload['filename']  = os.path.basename(filepath)
    R.payload['metadata']  = file_metadata(filepath)

    R.pdf_text(f'Deepfake Detection Report — Job {job_id}', 'Title')
    R.pdf_text(
        f'File: {os.path.basename(filepath)}  |  '
        f'Type: {file_type}  |  '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    )

    if file_type == 'UNSUPPORTED':
        R.payload['error'] = f'Unsupported file format: {ext}'
        return R.payload

    try:
        if   file_type == 'IMAGE'    : final_score = analyze_image(filepath, R)
        elif file_type == 'VIDEO'    : final_score = analyze_video(filepath, R)
        elif file_type == 'AUDIO'    : final_score = analyze_audio(filepath, R)
        elif file_type == 'DOCUMENT' : final_score = analyze_document(filepath, R)

        # ── Non-image modalities: legacy behaviour unchanged ──────────────────
        if file_type != 'IMAGE':
            R.payload['final_score']  = round(final_score, 1)
            R.payload['threat_level'] = threat_from_score(final_score)
            R.payload['verdict']      = verdict_text(final_score)

        # ── IMAGE: dominant classification + indicator filtering ──────────────
        else:
            # classify_dominant() reads stage_scores already written by
            # analyze_image() and returns new fields to merge in.
            clf = classify_dominant(R.payload)

            # Preserve raw indicator list for debugging / PDF
            R.payload['all_indicators'] = list(R.payload.get('indicators', []))

            # Filter indicators to only those supporting the dominant class
            # AND appropriate for the content type (face vs vehicle)
            R.payload['indicators'] = filter_indicators(
                R.payload.get('indicators', []),
                clf['classification'],
                has_human_face=R.payload.get('has_human_face', True),
            )

            # Merge classification fields into payload
            # (final_score and threat_level are overwritten with mapped values)
            R.payload.update(clf)

    except Exception as e:
        tb = traceback.format_exc()
        R.payload['error'] = str(e)
        R.pdf_text(f'PIPELINE ERROR: {e}')
        R.pdf_text(tb)
        print(f'Pipeline error: {e}\n{tb}')

    # Always try to build the PDF
    try:
        R.build_pdf()
        R.payload['pdf_ready'] = True
    except Exception as e:
        R.payload['pdf_ready'] = False
        print(f'PDF build error: {e}')

    return R.payload
