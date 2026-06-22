"""
Configuration — folder paths, constants.
Replaces notebook Cell 3 (Colab-specific /content/ paths removed,
now uses a proper temp directory suitable for any Linux host).
"""
import os
import tempfile

# Use the system temp dir as a base — works identically on Railway,
# local dev, or any standard Linux container, unlike the old
# Colab-specific '/content/dfd_tmp' paths.
BASE_TMP_DIR = os.environ.get("DFD_TMP_DIR", os.path.join(tempfile.gettempdir(), "dfd"))

TMP_FOLDER    = os.path.join(BASE_TMP_DIR, "uploads")
REPORT_FOLDER = os.path.join(BASE_TMP_DIR, "reports")

os.makedirs(TMP_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Max upload size (bytes) — 500MB, same as the original Colab server
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

# CORS — restrict this to your real frontend domain in production
# (the Colab prototype used '*' for convenience; tighten before launch)
ALLOWED_ORIGINS = os.environ.get("DFD_ALLOWED_ORIGINS", "*").split(",")

IMAGE_FORMATS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".mpo"}
VIDEO_FORMATS    = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".wmv"}
AUDIO_FORMATS    = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"}
DOCUMENT_FORMATS = {".txt", ".pdf", ".doc", ".docx", ".rtf", ".md"}
