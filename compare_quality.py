#!/usr/bin/env python3
"""
Video Quality Comparison Tool
Compares baseline vs optimized video encoding quality.
"""

import os
import argparse
import video_utils


def compare_videos(baseline_path, optimized_path, source_pattern, output_dir="."):
    """Compare two encoded videos against source."""
    print("=" * 70)
    print(f"  COMPARISON: {os.path.basename(baseline_path)} vs {os.path.basename(optimized_path)}")
    print("=" * 70)

    if not os.path.exists(baseline_path) or not os.path.exists(optimized_path):
        print("❌ Input files not found.")
        return

    # File sizes
    base_size = video_utils.get_file_size_mb(baseline_path)
    opt_size = video_utils.get_file_size_mb(optimized_path)
    print(f"Sizes: {base_size:.2f} MB vs {opt_size:.2f} MB")
    print(f"Diff:  {opt_size - base_size:+.2f} MB ({((opt_size / base_size - 1) * 100):+.1f}%)")

    # Bitrate variance
    print("\n📊 Bitrate Analysis...")
    base_stats = video_utils.analyze_bitrate_distribution(baseline_path)
    opt_stats = video_utils.analyze_bitrate_distribution(optimized_path)

    if base_stats and opt_stats:
        print(f"   Std Dev: {base_stats['stdev'] / 1024:.2f} KB vs {opt_stats['stdev'] / 1024:.2f} KB")
        var_diff = (opt_stats['stdev'] / base_stats['stdev'] - 1) * 100
        print(f"   Variance Change: {var_diff:+.1f}%")

    # Reference video
    print("\n📹 Generating Reference...")
    ref_video = os.path.join(output_dir, "temp_reference.mp4")
    video_utils.create_reference_video(source_pattern, ref_video)

    # Quality metrics
    print("\n🔍 Quality Metrics (vs Source)...")
    p_base = video_utils.calculate_psnr(ref_video, baseline_path)
    s_base = video_utils.calculate_ssim(ref_video, baseline_path)
    print(f"   Baseline:  PSNR {p_base:.2f} dB | SSIM {s_base:.4f}")

    p_opt = video_utils.calculate_psnr(ref_video, optimized_path)
    s_opt = video_utils.calculate_ssim(ref_video, optimized_path)
    print(f"   Optimized: PSNR {p_opt:.2f} dB | SSIM {s_opt:.4f}")

    # Visual comparison frames
    print("\n🖼️  Extracting Comparison Frames...")
    frames = [50, 150, 250, 350, 450]
    for f in frames:
        b_img = os.path.join(output_dir, f"_tmp_base_{f}.png")
        o_img = os.path.join(output_dir, f"_tmp_opt_{f}.png")
        comp_img = os.path.join(output_dir, f"compare_{f}.png")

        video_utils.extract_frame(baseline_path, f, b_img)
        video_utils.extract_frame(optimized_path, f, o_img)
        video_utils.create_comparison_image(b_img, o_img, comp_img)
        print(f"   Saved {comp_img}")

        # Cleanup temp files
        for tmp in [b_img, o_img]:
            if os.path.exists(tmp):
                os.remove(tmp)

    # Cleanup reference
    if os.path.exists(ref_video):
        os.remove(ref_video)
    print("\n✅ Comparison Complete.")

def main():
    parser = argparse.ArgumentParser(description="Compare Video Quality")
    parser.add_argument('--baseline', default="baseline.mp4")
    parser.add_argument('--optimized', default="optimized.mp4")
    parser.add_argument('--source', default="./Mandelbulb_Dataset/output_color/frame_%03d.png")
    parser.add_argument('--output', default=".")

    args = parser.parse_args()
    compare_videos(args.baseline, args.optimized, args.source, args.output)


if __name__ == "__main__":
    main()
