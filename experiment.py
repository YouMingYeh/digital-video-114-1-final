#!/usr/bin/env python3
"""
ROI Quality Experiment
Demonstrates that Adaptive Quantization shifts quality from background to foreground.

Key Finding: AQ mode 3 with strength 4.0 shifts ~5-6 dB from background to ROI.
"""

import os
import subprocess
import video_utils

# Configuration - use relative paths
COLOR_DIR = "./Mandelbulb_Dataset/output_color"
FRAME_PATTERN = f"{COLOR_DIR}/frame_%03d.png"

# Crop filters for region analysis
ROI_CROP = "crop=1024:1024:512:512"      # Center (Mandelbulb)
BACKGROUND_CROP = "crop=512:512:0:0"      # Top-left corner (black background)


def encode_test(name, crf, x264_params=None):
    """Encode a test video and return its path."""
    output = f"test_{name}.mp4"
    cmd = [
        'ffmpeg', '-y', '-v', 'quiet', '-stats',
        '-framerate', '30',
        '-i', FRAME_PATTERN,
        '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf),
    ]
    if x264_params:
        cmd.extend(['-x264-params', x264_params])
    cmd.extend(['-pix_fmt', 'yuv420p', output])
    subprocess.run(cmd)
    return output


def measure_region_quality(video_path, ref_video):
    """Measure PSNR for global, ROI, and background regions."""
    global_psnr = video_utils.calculate_psnr(ref_video, video_path)
    roi_psnr = video_utils.calculate_psnr(ref_video, video_path, crop_filter=ROI_CROP)
    bg_psnr = video_utils.calculate_psnr(ref_video, video_path, crop_filter=BACKGROUND_CROP)
    return global_psnr, roi_psnr, bg_psnr


def run_experiment():
    """Run the quality redistribution experiment."""
    print("=" * 80)
    print("🎯 ROI Quality Redistribution Experiment")
    print("=" * 80)

    # Create lossless reference
    ref_video = "test_reference.mp4"
    print("\n📹 Creating reference video...")
    video_utils.create_reference_video(FRAME_PATTERN, ref_video)

    # Test configurations
    configs = [
        ("Baseline", None),
        ("AQ 2.0", "aq-mode=3:aq-strength=2.0"),
        ("AQ 4.0", "aq-mode=3:aq-strength=4.0"),
    ]
    crfs = [23, 28, 33, 38]

    # Header
    print("\n" + "=" * 80)
    print(f"{'CRF':<6} {'Config':<12} {'Size MB':<10} {'Global':<10} {'ROI':<10} {'Background':<12} {'ROI-BG':<10}")
    print("-" * 80)

    summary = []

    for crf in crfs:
        print(f"\n📊 CRF {crf}:")
        baseline_diff = None

        for name, params in configs:
            # Encode
            path = encode_test(f"{name.lower().replace(' ', '_')}_{crf}", crf, params)
            size = video_utils.get_file_size_mb(path)

            # Measure
            global_psnr, roi_psnr, bg_psnr = measure_region_quality(path, ref_video)
            diff = roi_psnr - bg_psnr

            if baseline_diff is None:
                baseline_diff = diff

            shift = diff - baseline_diff
            print(f"{crf:<6} {name:<12} {size:<10.2f} {global_psnr:<10.2f} {roi_psnr:<10.2f} {bg_psnr:<12.2f} {diff:<+10.2f}")

            # Store for summary
            if name == "AQ 4.0":
                summary.append((crf, shift))

            # Cleanup test file
            if os.path.exists(path):
                os.remove(path)

    # Summary
    print("\n" + "=" * 80)
    print("📈 KEY RESULTS: Quality Shifted from Background to ROI")
    print("=" * 80)

    for crf, shift in summary:
        status = "✅" if shift > 0 else "❌"
        print(f"  CRF {crf}: {status} {shift:+.2f} dB shifted to ROI")

    avg_shift = sum(s for _, s in summary) / len(summary)
    print(f"\n  Average: {avg_shift:+.2f} dB quality redistribution")

    # Cleanup reference
    if os.path.exists(ref_video):
        os.remove(ref_video)


if __name__ == "__main__":
    run_experiment()
