"""
Main pipeline dispatcher — routes each uploaded file to the correct
modality-specific analysis function and assembles the final JSON payload.
Replaces notebook Cell 13 (run_pipeline).
"""
import os
import traceback
from datetime import datetime

from .config import IMAGE_FORMATS, VIDEO_FORMATS, AUDIO_FORMATS, DOCUMENT_FORMATS, REPORT_FOLDER
from .result import AnalysisResult
from .helpers import file_metadata, threat_from_score, verdict_text
from .image_pipeline import analyze_image
from .video_pipeline import analyze_video
from .audio_pipeline import analyze_audio
from .document_pipeline import analyze_document

import psutil

def mem():
    process = psutil.Process()
    print(
        f"MEMORY: {process.memory_info().rss / 1024 / 1024:.1f} MB"
    )
    
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

    # PDF header
    R.pdf_text(f'Deepfake Detection Report — Job {job_id}', 'Title')
    R.pdf_text(f'File: {os.path.basename(filepath)}  |  Type: {file_type}  |  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    if file_type == 'UNSUPPORTED':
        R.payload['error'] = f'Unsupported file format: {ext}'
        return R.payload

    try:
        print("BEFORE IMAGE PIPELINE")
        mem()

        if   file_type == 'IMAGE'    : final_score = analyze_image(filepath, R)
        elif file_type == 'VIDEO'    : final_score = analyze_video(filepath, R)
        elif file_type == 'AUDIO'    : final_score = analyze_audio(filepath, R)
        elif file_type == 'DOCUMENT' : final_score = analyze_document(filepath, R)
        print("AFTER IMAGE PIPELINE")
        mem()

        R.payload['final_score']  = round(final_score, 1)
        R.payload['threat_level'] = threat_from_score(final_score)
        R.payload['verdict']      = verdict_text(final_score)

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

    # NOTE: do NOT delete R.tmp_dir here — graph PNG files inside it
    # are served individually via GET /graph/{job_id}/{filename}.
    # They will be overwritten naturally on the next job that uses the same job_id.

    return R.payload
