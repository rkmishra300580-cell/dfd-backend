"""
detector/evaluate.py
Darpan  -  Labeled-set calibration harness

Runs the REAL production pipeline (analyze_image() + classify_dominant())
against the 9 labeled evidence images reconstructed from the 3 Aug / 4 Aug
handoffs, and reports per-image predictions vs ground truth. This is the
required step before touching any threshold in helpers.py
(CORROBORATION_DL_AI_THRESHOLD, NO_EXIF_CORROBORATED_AI_SCORE,
AI_GEN_THRESHOLD, DL_AI_FLOOR, VEHICLE_FLOOR) — none of those get changed
based on this script's own output; that's a separate decision for whoever
reviews the results.

--------------------------------------------------------------------------
IMPORTANT — environment
--------------------------------------------------------------------------
This script has been syntax-checked (py_compile) and reviewed line-by-line
against the real analyze_image() / classify_dominant() signatures and the
real AnalysisResult.payload contract (all three read from the actual
uploaded source, not reconstructed from comments). It has NOT been
executed end-to-end anywhere. It requires torch + transformers (for the
three HF models loaded inside image_pipeline.py) and network access to
huggingface.co — neither is available in the dev sandbox this was written
in, so the first real run of this script IS the calibration step itself,
not a pre-verified formality.

--------------------------------------------------------------------------
Usage (run inside the backend repo, so relative imports resolve):
    python -m detector.evaluate --images-dir /path/to/images

Each image must be present under --images-dir using its ORIGINAL analysed
filename. These are copied verbatim from each PDF report's header/footer
("File analysed: ...") — the literal filenames the pipeline saw in
production — not reconstructed or guessed:

    3607579ddbce_52BC7DB8-4E25-4E13-AE44-F3C4DBAC37D5.PNG
    5fd820318998_1A27DB72-9905-4B7D-BDA0-A0C8AB20E30D.PNG
    125a7f07ae17_WhatsApp_Image_2026-07-14_at_21.54.17__1_.jpeg
    f64fe20b9c3e_WhatsApp_Image_2026-07-14_at_21.50.08.jpeg
    f3819f373b0c_WhatsApp_Image_2026-07-14_at_21.54.17.jpeg
    552651bb58c5_WhatsApp_Image_2026-07-14_at_21.54.18.jpeg
    f5e59f773f15_WhatsApp_Image_2026-07-14_at_21.44.24.jpeg
    bdfd09aadb9a_WhatsApp_Image_2026-07-14_at_21.47.55.jpeg
    d79232d43480_WhatsApp_Image_2026-07-14_at_21.53.20.jpeg

--json-out PATH additionally dumps full per-image results, including the
complete stage_scores dict for every image, so the specific weighted
components (dl_ai, freq, vehicle_val, exif_ai, etc.) behind each score can
be inspected directly rather than re-derived from the headline number.
"""
import argparse
import json
import sys
from pathlib import Path

from .result import AnalysisResult
from .image_pipeline import analyze_image
from .helpers import classify_dominant


# Ground truth + prior (3 Aug baseline, confirmed against these same 4 Aug
# PDF reports) scores. Reconstructed by extracting each report's final
# score and matching against the 3 Aug handoff's stated tally (5/7 vehicle
# hits + 2 FPs, 1/2 face hits) — confirmed an exact match, so this is the
# same evidence batch that produced the original bug reports, not a fresh
# one.
MANIFEST = [
    {
        "report_id": "d79232d43480",
        "filename": "d79232d43480_WhatsApp_Image_2026-07-14_at_21.53.20.jpeg",
        "ground_truth": "REAL",
        "content_type": "VEHICLE",
        "prior_final_score": 38.7,
        "prior_outcome": "correct",
    },
    {
        "report_id": "bdfd09aadb9a",
        "filename": "bdfd09aadb9a_WhatsApp_Image_2026-07-14_at_21.47.55.jpeg",
        "ground_truth": "REAL",
        "content_type": "VEHICLE",
        "prior_final_score": 58.4,
        "prior_outcome": "false_positive",
    },
    {
        "report_id": "f5e59f773f15",
        "filename": "f5e59f773f15_WhatsApp_Image_2026-07-14_at_21.44.24.jpeg",
        "ground_truth": "REAL",
        "content_type": "VEHICLE",
        "prior_final_score": 98.7,
        "prior_outcome": "false_positive_severe",
    },
    {
        "report_id": "125a7f07ae17",
        "filename": "125a7f07ae17_WhatsApp_Image_2026-07-14_at_21.54.17__1_.jpeg",
        "ground_truth": "REAL",
        "content_type": "VEHICLE",
        "prior_final_score": 44.7,
        "prior_outcome": "correct_thin_margin",
    },
    {
        "report_id": "f64fe20b9c3e",
        "filename": "f64fe20b9c3e_WhatsApp_Image_2026-07-14_at_21.50.08.jpeg",
        "ground_truth": "REAL",
        "content_type": "VEHICLE",
        "prior_final_score": 35.6,
        "prior_outcome": "correct",
    },
    {
        "report_id": "f3819f373b0c",
        "filename": "f3819f373b0c_WhatsApp_Image_2026-07-14_at_21.54.17.jpeg",
        "ground_truth": "REAL",
        "content_type": "VEHICLE",
        "prior_final_score": 40.8,
        "prior_outcome": "correct",
    },
    {
        "report_id": "552651bb58c5",
        "filename": "552651bb58c5_WhatsApp_Image_2026-07-14_at_21.54.18.jpeg",
        "ground_truth": "REAL",
        "content_type": "VEHICLE",
        "prior_final_score": 44.4,
        "prior_outcome": "correct_thin_margin",
    },
    {
        "report_id": "3607579ddbce",
        "filename": "3607579ddbce_52BC7DB8-4E25-4E13-AE44-F3C4DBAC37D5.PNG",
        "ground_truth": "AI_GENERATED",
        "content_type": "FACE",
        "prior_final_score": 97.6,
        "prior_outcome": "hit",
    },
    {
        "report_id": "5fd820318998",
        "filename": "5fd820318998_1A27DB72-9905-4B7D-BDA0-A0C8AB20E30D.PNG",
        "ground_truth": "AI_GENERATED",
        "content_type": "FACE",
        "prior_final_score": 31.4,
        "prior_outcome": "miss",
    },
]


def evaluate_one(images_dir: Path, entry: dict) -> dict:
    """
    Runs the real pipeline on one labeled image and returns the ground
    truth entry merged with the fresh prediction. Never raises — errors
    (missing file, pipeline exception) are captured in the returned dict
    so one bad image doesn't abort the rest of the batch.
    """
    filepath = images_dir / entry["filename"]
    if not filepath.exists():
        return {**entry, "error": f"image not found: {filepath}"}

    R = AnalysisResult(entry["report_id"])
    # Set explicitly rather than relying on classify_dominant()'s internal
    # default (file_type defaults to 'IMAGE' if absent) or on filename
    # ever getting set by analyze_image()/result.py themselves — confirmed
    # by reading both files that neither ever writes R.payload['filename']
    # or R.payload['file_type']; that's done by the real app's request
    # handler, which this script has no access to and isn't part of the
    # detector package.
    R.payload["file_type"] = "IMAGE"
    R.payload["filename"] = entry["filename"]

    try:
        analyze_image(str(filepath), R)
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see docstring
        return {**entry, "error": f"analyze_image() raised: {exc!r}"}

    # analyze_image() does NOT call classify_dominant() itself - confirmed
    # by reading image_pipeline.py (the comment above dl_detector()'s
    # fusion-mode block: "... always calls classify_dominant() after
    # analyze_image() returns"). In production that merge happens in the
    # app's request handler, outside this package, so evaluate.py has to
    # do it explicitly to see the real, final classification rather than
    # dl_detector()'s intermediate return value.
    try:
        classification_fields = classify_dominant(R.payload)
    except Exception as exc:  # noqa: BLE001
        return {**entry, "error": f"classify_dominant() raised: {exc!r}"}
    R.payload.update(classification_fields)

    predicted = R.payload.get("classification", "UNKNOWN")
    gt_synthetic = entry["ground_truth"] != "REAL"
    pred_synthetic = predicted not in ("REAL", "UNKNOWN")
    correct = (gt_synthetic == pred_synthetic) if predicted != "UNKNOWN" else None

    return {
        **entry,
        "error": None,
        "predicted_classification": predicted,
        "predicted_final_score": R.payload.get("final_score"),
        "predicted_dominant_label": R.payload.get("dominant_label"),
        "predicted_risk_level": R.payload.get("risk_level"),
        "predicted_manipulation_type": R.payload.get("manipulation_type"),
        "stage_scores": R.payload.get("stage_scores", {}),
        "correct_real_vs_synthetic": correct,
    }


def _print_summary(results: list[dict]) -> None:
    header = f"{'report_id':<14} {'ground_truth':<13} {'prior':>7} {'new':>7} {'predicted':<14} {'match':<5}"
    print(header)
    print("-" * len(header))

    evaluated, correct_count, errored = 0, 0, 0
    for r in results:
        if r.get("error"):
            errored += 1
            print(f"{r['report_id']:<14} ERROR: {r['error']}")
            continue
        evaluated += 1
        match = r["correct_real_vs_synthetic"]
        match_str = "YES" if match else ("NO" if match is False else "?")
        if match:
            correct_count += 1
        prior = f"{r['prior_final_score']:.1f}%"
        new = f"{r['predicted_final_score']:.1f}%" if r["predicted_final_score"] is not None else "n/a"
        print(f"{r['report_id']:<14} {r['ground_truth']:<13} {prior:>7} {new:>7} "
              f"{r['predicted_classification']:<14} {match_str:<5}")

    print("-" * len(header))
    if evaluated:
        print(f"{correct_count}/{evaluated} correct (REAL vs synthetic)"
              + (f", {errored} image(s) errored/missing" if errored else ""))
    else:
        print("No images could be evaluated — check --images-dir and the filenames listed "
              "in this script's module docstring.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the real Darpan pipeline against the 9 labeled evidence images."
    )
    ap.add_argument("--images-dir", required=True, type=Path,
                     help="Directory containing the 9 images, named with their original filenames.")
    ap.add_argument("--json-out", type=Path, default=None,
                     help="Optional path to write full per-image results (including stage_scores) as JSON.")
    args = ap.parse_args()

    if not args.images_dir.is_dir():
        print(f"ERROR: {args.images_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    results = [evaluate_one(args.images_dir, entry) for entry in MANIFEST]
    _print_summary(results)

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nFull results (including stage_scores) written to {args.json_out}")


if __name__ == "__main__":
    main()
