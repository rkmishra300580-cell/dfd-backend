"""
uploads_guard.py — server-side file size + MIME validation.

FIXED (was reading the entire file into memory before checking size at all —
same bug existed in v5.2's main.py, just moved here unchanged in the first pass):
now reads in fixed-size chunks and aborts as soon as the running total crosses
max_bytes, so an oversized/malicious upload never gets fully buffered in RAM.
Worst-case memory for a REJECTED file is ~max_bytes + one chunk, not the full
upload size — this matters on a 2GB Render instance with a 500MB cap.

A Content-Length pre-check is also done first as a fast, near-free rejection
for obviously oversized requests, before any chunk reading starts. It's an
upper-bound estimate (Content-Length covers the whole multipart body, not just
this one file, so it can slightly overstate the file's real size) — the
chunked read below is what actually enforces the limit precisely.
"""
import magic
from fastapi import UploadFile, HTTPException, Request

CHUNK_SIZE = 1024 * 1024  # 1MB — small enough to bound memory, large enough to not be slow

ALLOWED_MIME_TYPES = {
    # IMAGE - matches config.py's IMAGE_FORMATS
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff",

    # AUDIO - matches config.py's AUDIO_FORMATS ({.mp3, .wav, .aac, .flac, .ogg, .m4a})
    # BUG FIX (26 Aug 2026): this allowlist was image-only - the comment
    # below said "extend here once video/audio/document pipelines are
    # audited and wired up", but audio_pipeline.py/video_pipeline.py/
    # document_pipeline.py were already live and dispatched to from
    # pipeline.py. Every non-image upload was being rejected with a 415
    # before ever reaching the detector package - confirmed live: a
    # genuinely valid WAV file (RIFF/WAVE, PCM 16-bit stereo, verified
    # with `file` against the actual header bytes, not just the
    # extension) was rejected in production. Same root cause as main.py's
    # hardcoded modality='image' - the API layer was never updated to
    # match what the pipeline layer already supported.
    "audio/mpeg",                # .mp3
    "audio/x-wav", "audio/wav",  # .wav - libmagic reports either depending on version; both included
    "audio/aac", "audio/x-hx-aac-adts",  # .aac - container-dependent detection
    "audio/x-flac", "audio/flac",        # .flac - libmagic reports either depending on version
    "audio/ogg",                 # .ogg
    "audio/mp4", "audio/x-m4a",  # .m4a - MP4-container audio, detection varies

    # VIDEO - matches config.py's VIDEO_FORMATS
    "video/mp4",
    "video/x-msvideo",   # .avi
    "video/quicktime",   # .mov
    "video/x-matroska",  # .mkv
    "video/x-flv",       # .flv
    "video/webm",
    "video/x-ms-wmv",    # .wmv

    # DOCUMENT - matches config.py's DOCUMENT_FORMATS
    "text/plain",         # .txt, .md (markdown has no distinct binary signature)
    "application/pdf",
    "application/msword", # .doc
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/zip",    # .docx is a zip container - some libmagic versions/builds report this
                           # instead of the docx-specific MIME above; included so a valid .docx
                           # isn't rejected on a build that detects it this way
    "text/rtf", "application/rtf",  # .rtf - libmagic reports either depending on version
}

# CAVEAT (26 Aug 2026): the audio/video/document MIME strings above are
# standard/commonly-documented libmagic output for these formats. One was
# directly verified, not just documented: audio/x-wav was confirmed by
# running magic.from_buffer() against the exact real file that produced
# the production 415 (a genuine RIFF/WAVE PCM file) - it detected as
# audio/x-wav precisely, confirming this fix resolves that real case. The
# rest (other audio formats, video, document) were not individually
# tested against real files in this environment. If a valid upload of a
# supported format still gets rejected, the error response's own detail
# field states exactly what was detected (see the 415 raise below) -
# check Render's logs for that specific string and add it here rather
# than guessing further.


async def validate_upload(file: UploadFile, max_bytes: int, request: Request = None) -> bytes:
    if request is not None:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > max_bytes:
                    raise HTTPException(status_code=413, detail=f"File exceeds {max_bytes} byte limit")
            except ValueError:
                pass  # malformed header — fall through to the authoritative chunked check below

    chunks = []
    total = 0
    first_chunk = b""

    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            # Stop reading immediately — do not keep buffering an oversized upload.
            raise HTTPException(status_code=413, detail=f"File exceeds {max_bytes} byte limit")
        if not first_chunk:
            first_chunk = chunk
        chunks.append(chunk)

    contents = b"".join(chunks)

    # MIME sniffing only needs the first chunk — no need to re-read the full buffer.
    detected_mime = magic.from_buffer(first_chunk, mime=True)
    if detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type detected: {detected_mime} "
                   f"(this is sniffed from file content, not the extension)",
        )

    return contents
