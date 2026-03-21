"""
Run the full SAM3 + DINOv2 shelf product matching pipeline.

Usage:
    python run_pipeline.py                   # run all steps
    python run_pipeline.py --step 1          # run only step 1
    python run_pipeline.py --step 3,4        # run steps 3 and 4
    python run_pipeline.py --max-images 5    # limit to 5 shelf images (for testing)
"""

import argparse
import time

import config


def main():
    parser = argparse.ArgumentParser(description="SAM3 + DINOv2 product matching pipeline")
    parser.add_argument("--step", type=str, default=None,
                        help="Comma-separated step numbers to run (default: all)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit shelf images processed (default: all)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override similarity threshold")
    args = parser.parse_args()

    if args.max_images:
        config.MAX_IMAGES = args.max_images
    if args.threshold:
        config.SIMILARITY_THRESHOLD = args.threshold

    steps_to_run = set()
    if args.step:
        steps_to_run = {int(s.strip()) for s in args.step.split(",")}
    else:
        steps_to_run = {1, 2, 3, 4}

    config.ensure_dirs()
    total_start = time.time()

    # ── Step 1: Segment with SAM3 ──────────────────────────────────────
    if 1 in steps_to_run:
        print("=" * 60)
        print("STEP 1: Segmenting shelf images with SAM3")
        print("=" * 60)
        t = time.time()
        import step1_segment
        step1_segment.run(max_images=config.MAX_IMAGES)
        print(f"Step 1 completed in {time.time() - t:.1f}s\n")

    # ── Step 2: Masks → bounding squares ───────────────────────────────
    if 2 in steps_to_run:
        print("=" * 60)
        print("STEP 2: Converting masks to bounding squares")
        print("=" * 60)
        t = time.time()
        import step2_squares
        step2_squares.run()
        print(f"Step 2 completed in {time.time() - t:.1f}s\n")

    # ── Step 3: Pre-embed product images ───────────────────────────────
    if 3 in steps_to_run:
        print("=" * 60)
        print("STEP 3: Embedding product images with DINOv2")
        print("=" * 60)
        t = time.time()
        import step3_embed_products
        step3_embed_products.run()
        print(f"Step 3 completed in {time.time() - t:.1f}s\n")

    # ── Step 4: Crop, embed, match, filter ─────────────────────────────
    if 4 in steps_to_run:
        print("=" * 60)
        print("STEP 4: Matching shelf crops against product database")
        print("=" * 60)
        t = time.time()
        import step4_match
        step4_match.run()
        print(f"Step 4 completed in {time.time() - t:.1f}s\n")

    total_time = time.time() - total_start
    print("=" * 60)
    print(f"Pipeline complete! Total time: {total_time:.1f}s")
    print(f"Check outputs in: {config.OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
