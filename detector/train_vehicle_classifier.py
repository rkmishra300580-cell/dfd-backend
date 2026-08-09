"""
detector/train_vehicle_classifier.py

Standalone CLI - run manually once you have labeled vehicle-domain images
for BOTH classes (real vehicle photos AND AI-generated vehicle photos).

NOT imported anywhere in the production request path (pipeline.py never
imports this file) - this keeps sklearn/joblib and the training-only code
entirely out of the request-serving process. score_vehicle_domain() in
vehicle_ai_gen_classifier.py is the only thing the live server touches,
and it only ever reads the .joblib file this script produces.

Usage (on Render Shell or locally):
    python -m detector.train_vehicle_classifier \\
        --real-dir /path/to/real_vehicle_photos \\
        --ai-dir   /path/to/ai_generated_vehicle_photos

Each directory should contain image files directly (jpg/jpeg/png/etc,
no subfolders). Prints real held-out accuracy and a confusion matrix
before saving anything - read that output before trusting the model,
same discipline as evaluate.py for the main pipeline's thresholds.
"""
import argparse
import os

from .vehicle_ai_gen_classifier import train_and_save

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')


def _list_images(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Not a directory: {directory}')
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith(IMAGE_EXTS)
    ]


def main():
    parser = argparse.ArgumentParser(
        description='Train the vehicle-domain AI-generation classifier from labeled image folders.'
    )
    parser.add_argument('--real-dir', required=True,
                         help='Directory of ground-truth REAL vehicle photos')
    parser.add_argument('--ai-dir', required=True,
                         help='Directory of ground-truth AI-generated vehicle photos')
    args = parser.parse_args()

    real_paths = _list_images(args.real_dir)
    ai_paths   = _list_images(args.ai_dir)

    print(f'Found {len(real_paths)} REAL images in {args.real_dir}')
    print(f'Found {len(ai_paths)} AI_GENERATED images in {args.ai_dir}')

    train_and_save(real_paths, ai_paths)


if __name__ == '__main__':
    main()
