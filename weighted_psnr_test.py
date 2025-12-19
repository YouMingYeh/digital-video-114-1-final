"""
Detailed Weighted PSNR Analysis

Tests the hypothesis: At the same file size, does preprocessing
give BETTER foreground quality at the cost of background?
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from PIL import Image

from config import OUTPUT_DIR, COLOR_FRAMES_DIR, DEPTH_FRAMES_DIR, DepthConfig
from utils import load_color_frame, load_depth_frame, depth_to_importance, compute_psnr
from method2_depth_quality import apply_depth_quality_bilateral, apply_depth_quality_blur


def compute_weighted_psnr(orig: np.ndarray, dec: np.ndarray, importance: np.ndarray, fg_weight: float = 3.0) -> float:
    """Weighted PSNR: foreground pixels count more."""
    weights = 1.0 + (fg_weight - 1.0) * importance
    weights = weights / weights.mean()
    if orig.ndim == 3:
        weights = np.expand_dims(weights, axis=-1)
    diff = (orig.astype(float) - dec.astype(float)) ** 2
    weighted_mse = np.sum(weights * diff) / np.sum(weights * np.ones_like(diff))
    if weighted_mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(weighted_mse))


def encode_to_target_size(frames_dir: Path, output_path: Path, start_frame: int,
                          num_frames: int, target_size_mb: float) -> tuple[Path, float, int]:
    """Encode to target size using CRF binary search."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crf_low, crf_high = 15, 51
    best_crf, best_size = 28, 999

    for _ in range(10):
        crf = (crf_low + crf_high) // 2
        cmd = [
            "ffmpeg", "-y", "-framerate", "30",
            "-start_number", str(start_frame),
            "-i", str(frames_dir / "frame_%03d.png"),
            "-frames:v", str(num_frames),
            "-c:v", "libx264", "-crf", str(crf),
            "-preset", "medium", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
        size_mb = output_path.stat().st_size / (1024 * 1024)

        if abs(size_mb - target_size_mb) < abs(best_size - target_size_mb):
            best_crf, best_size = crf, size_mb
        if abs(size_mb - target_size_mb) / target_size_mb < 0.05:
            return output_path, size_mb, crf
        if size_mb > target_size_mb:
            crf_low = crf
        else:
            crf_high = crf

    return output_path, best_size, best_crf


def preprocess_frames(start_frame: int, num_frames: int, output_dir: Path, method: str) -> Path:
    """Preprocess frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_config = DepthConfig()
    for i in range(num_frames):
        idx = start_frame + i
        color = load_color_frame(idx)
        depth = load_depth_frame(idx)
        if method == "bilateral":
            processed = apply_depth_quality_bilateral(color, depth, depth_config)
        else:
            processed = apply_depth_quality_blur(color, depth, depth_config)
        Image.fromarray(processed).save(output_dir / f"frame_{idx:03d}.png")
    return output_dir


def compute_all_metrics(video_path: Path, original_dir: Path, start_frame: int, num_frames: int) -> dict:
    """Decode and compute all metrics."""
    # Decode
    decoded_dir = video_path.parent / "decoded"
    decoded_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video_path), str(decoded_dir / "frame_%03d.png")]
    subprocess.run(cmd, capture_output=True)

    # Compute metrics
    overall_psnr, weighted_psnr = [], []
    roi_psnr, bg_psnr = [], []

    for i in range(num_frames):
        orig_idx = start_frame + i
        dec_idx = i + 1

        orig_path = original_dir / f"frame_{orig_idx:03d}.png"
        dec_path = decoded_dir / f"frame_{dec_idx:03d}.png"

        if not dec_path.exists():
            continue

        orig = np.array(Image.open(orig_path).convert("RGB"))
        dec = np.array(Image.open(dec_path).convert("RGB"))
        depth = load_depth_frame(orig_idx)
        importance = depth_to_importance(depth, "near_high")

        # Overall PSNR
        overall_psnr.append(compute_psnr(orig, dec))

        # Weighted PSNR (foreground 3x weight)
        weighted_psnr.append(compute_weighted_psnr(orig, dec, importance, 3.0))

        # ROI PSNR (foreground only, importance > 0.5)
        roi_mask = importance > 0.5
        if roi_mask.any():
            roi_psnr.append(compute_psnr(orig[roi_mask], dec[roi_mask]))

        # Background PSNR (importance < 0.3)
        bg_mask = importance < 0.3
        if bg_mask.any():
            bg_psnr.append(compute_psnr(orig[bg_mask], dec[bg_mask]))

    return {
        'overall_psnr': np.mean(overall_psnr) if overall_psnr else 0,
        'weighted_psnr': np.mean(weighted_psnr) if weighted_psnr else 0,
        'roi_psnr': np.mean(roi_psnr) if roi_psnr else 0,
        'bg_psnr': np.mean(bg_psnr) if bg_psnr else 0,
    }


def run_analysis(target_size_mb: float, segment: str = "rotation", num_frames: int = 150):
    """Run weighted PSNR analysis at target size."""
    start_frame = 0 if segment == "rotation" else 300
    work_dir = OUTPUT_DIR / f"weighted_analysis_{segment}_{target_size_mb}mb"
    work_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print(f"\n{'='*70}")
    print(f"WEIGHTED PSNR ANALYSIS: {target_size_mb} MB, {segment}")
    print(f"{'='*70}")

    # Baseline
    print("\n[1/3] Baseline...")
    baseline_path, baseline_size, baseline_crf = encode_to_target_size(
        COLOR_FRAMES_DIR, work_dir / "baseline.mp4", start_frame, num_frames, target_size_mb
    )
    baseline_metrics = compute_all_metrics(baseline_path, COLOR_FRAMES_DIR, start_frame, num_frames)
    results['baseline'] = {'size': baseline_size, 'crf': baseline_crf, **baseline_metrics}

    # Bilateral
    print("[2/3] Bilateral...")
    bilateral_frames = preprocess_frames(start_frame, num_frames, work_dir / "bilateral_frames", "bilateral")
    bilateral_path, bilateral_size, bilateral_crf = encode_to_target_size(
        bilateral_frames, work_dir / "bilateral.mp4", start_frame, num_frames, target_size_mb
    )
    bilateral_metrics = compute_all_metrics(bilateral_path, COLOR_FRAMES_DIR, start_frame, num_frames)
    results['bilateral'] = {'size': bilateral_size, 'crf': bilateral_crf, **bilateral_metrics}

    # Blur
    print("[3/3] Blur...")
    blur_frames = preprocess_frames(start_frame, num_frames, work_dir / "blur_frames", "blur")
    blur_path, blur_size, blur_crf = encode_to_target_size(
        blur_frames, work_dir / "blur.mp4", start_frame, num_frames, target_size_mb
    )
    blur_metrics = compute_all_metrics(blur_path, COLOR_FRAMES_DIR, start_frame, num_frames)
    results['blur'] = {'size': blur_size, 'crf': blur_crf, **blur_metrics}

    return results


def main():
    all_results = {}

    for target in [0.3, 0.5, 1.0, 2.0, 3.0]:
        results = run_analysis(target_size_mb=target, segment="rotation", num_frames=150)
        all_results[target] = results

    # Print comprehensive summary
    print("\n" + "=" * 90)
    print("COMPREHENSIVE WEIGHTED PSNR ANALYSIS")
    print("=" * 90)

    print("\n1. OVERALL PSNR (all pixels equal)")
    print("-" * 70)
    print(f"{'Target':<10} {'Baseline':>12} {'Bilateral':>12} {'Blur':>12} {'Winner':>12}")
    for target, r in all_results.items():
        b, bi, bl = r['baseline']['overall_psnr'], r['bilateral']['overall_psnr'], r['blur']['overall_psnr']
        winner = "Baseline" if b >= max(bi, bl) else ("Bilateral" if bi > bl else "Blur")
        print(f"{target} MB{'':<5} {b:>12.2f} {bi:>12.2f} {bl:>12.2f} {winner:>12}")

    print("\n2. WEIGHTED PSNR (foreground 3x weight)")
    print("-" * 70)
    print(f"{'Target':<10} {'Baseline':>12} {'Bilateral':>12} {'Blur':>12} {'Winner':>12}")
    for target, r in all_results.items():
        b, bi, bl = r['baseline']['weighted_psnr'], r['bilateral']['weighted_psnr'], r['blur']['weighted_psnr']
        winner = "Baseline" if b >= max(bi, bl) else ("Bilateral" if bi > bl else "Blur")
        print(f"{target} MB{'':<5} {b:>12.2f} {bi:>12.2f} {bl:>12.2f} {winner:>12}")

    print("\n3. ROI PSNR (foreground ONLY, importance > 0.5)")
    print("-" * 70)
    print(f"{'Target':<10} {'Baseline':>12} {'Bilateral':>12} {'Blur':>12} {'Winner':>12}")
    for target, r in all_results.items():
        b, bi, bl = r['baseline']['roi_psnr'], r['bilateral']['roi_psnr'], r['blur']['roi_psnr']
        winner = "Baseline" if b >= max(bi, bl) else ("Bilateral" if bi > bl else "Blur")
        print(f"{target} MB{'':<5} {b:>12.2f} {bi:>12.2f} {bl:>12.2f} {winner:>12}")

    print("\n4. BACKGROUND PSNR (importance < 0.3)")
    print("-" * 70)
    print(f"{'Target':<10} {'Baseline':>12} {'Bilateral':>12} {'Blur':>12} {'Winner':>12}")
    for target, r in all_results.items():
        b, bi, bl = r['baseline']['bg_psnr'], r['bilateral']['bg_psnr'], r['blur']['bg_psnr']
        winner = "Baseline" if b >= max(bi, bl) else ("Bilateral" if bi > bl else "Blur")
        print(f"{target} MB{'':<5} {b:>12.2f} {bi:>12.2f} {bl:>12.2f} {winner:>12}")

    print("\n" + "=" * 90)
    print("KEY QUESTION: Does preprocessing help FOREGROUND at cost of BACKGROUND?")
    print("=" * 90)
    print("\nIf ROI PSNR (foreground) for preprocessing > baseline, the hypothesis has merit!")

    # Save results
    with open(OUTPUT_DIR / "weighted_psnr_analysis.json", "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
