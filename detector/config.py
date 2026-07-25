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

# CORS — must NOT be '*' now that main.py sets allow_credentials=True (v6.0, for the
# Authorization header). A wildcard origin combined with allow_credentials is rejected
# outright by some Starlette/FastAPI versions, and even where it isn't, browsers will
# refuse to expose the response to credentialed cross-origin requests either way.
# Defaults to the real deployed frontend; override via DFD_ALLOWED_ORIGINS if that
# domain changes (comma-separated for multiple origins, e.g. staging + prod).
ALLOWED_ORIGINS = os.environ.get(
    "DFD_ALLOWED_ORIGINS", "https://veritas-deepfake-detector.vercel.app"
).split(",")

IMAGE_FORMATS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".mpo"}
VIDEO_FORMATS    = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".wmv"}
AUDIO_FORMATS    = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"}
DOCUMENT_FORMATS = {".txt", ".pdf", ".doc", ".docx", ".rtf", ".md"}
