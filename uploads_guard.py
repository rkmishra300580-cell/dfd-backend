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
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff",
    # extend here once video/audio/document pipelines are audited and wired up
}


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
