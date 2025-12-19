"""
FAIR perceptual comparison - ALL methods at SAME target size.
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile

from config import OUTPUT_DIR, COLOR_FRAMES_DIR, DEPTH_FRAMES_DIR, DepthConfig
from utils import load_color_frame, load_depth_frame, depth_to_importance, compute_psnr, compute_ssim
from method2_depth_quality import apply_depth_quality_bilateral, apply_depth_quality_blur


def compute_vmaf(reference_video: Path, distorted_video: Path) -> float:
    """Compute VMAF score."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            vmaf_log = f.name

        cmd = [
            "ffmpeg", "-y",
            "-i", str(distorted_video),
            "-i", str(reference_video),
            "-lavfi", f"libvmaf=log_fmt=json:log_path={vmaf_log}",
            "-f", "null", "-"
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)

        with open(vmaf_log) as f:
            vmaf_data = json.load(f)
            score = vmaf_data.get("pooled_metrics", {}).get("vmaf", {}).get("mean", 0)

        Path(vmaf_log).unlink(missing_ok=True)
        return score
    except Exception as e:
        print(f"VMAF error: {e}")
        return 0


def encode_to_target_size(frames_dir: Path, output_path: Path, start_frame: int,
                          num_frames: int, target_size_mb: float) -> tuple[Path, float, int]:
    """Encode to target size using CRF binary search. Returns path, actual size, crf used."""
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

        if abs(size_mb - target_size_mb) / target_size_mb < 0.05:  # Within 5%
            return output_path, size_mb, crf

        if size_mb > target_size_mb:
            crf_low = crf
        else:
            crf_high = crf

    # Re-encode with best CRF
    cmd[-4] = str(best_crf)
    subprocess.run(cmd, capture_output=True)
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


def decode_and_measure(video_path: Path, original_dir: Path, start_frame: int,
                       num_frames: int, work_dir: Path) -> dict:
    """Decode video and compute metrics."""
    decoded_dir = work_dir / "decoded"
    decoded_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", str(video_path), str(decoded_dir / "frame_%03d.png")]
    subprocess.run(cmd, capture_output=True)

    psnr_list, roi_psnr_list = [], []

    for i in range(num_frames):
        orig_idx = start_frame + i
        dec_idx = i + 1

        orig = np.array(Image.open(original_dir / f"frame_{orig_idx:03d}.png").convert("RGB"))
        dec_path = decoded_dir / f"frame_{dec_idx:03d}.png"
        if not dec_path.exists():
            continue
        dec = np.array(Image.open(dec_path).convert("RGB"))

        depth = load_depth_frame(orig_idx)
        importance = depth_to_importance(depth, "near_high")

        psnr_list.append(compute_psnr(orig, dec))

        roi_mask = importance > 0.5
        if roi_mask.any():
            roi_psnr_list.append(compute_psnr(orig[roi_mask], dec[roi_mask]))

    return {
        'psnr': np.mean(psnr_list) if psnr_list else 0,
        'roi_psnr': np.mean(roi_psnr_list) if roi_psnr_list else 0,
    }


def run_fair_comparison(target_size_mb: float, segment: str = "rotation", num_frames: int = 150):
    """Run fair comparison with ALL methods at SAME target size."""
    start_frame = 0 if segment == "rotation" else 300
    work_dir = OUTPUT_DIR / f"fair_perceptual_{segment}_{target_size_mb}mb"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"FAIR COMPARISON: {target_size_mb} MB target, {segment} segment")
    print(f"{'='*70}")

    # Create reference video for VMAF
    ref_path = work_dir / "reference.mp4"
    cmd = [
        "ffmpeg", "-y", "-framerate", "30",
        "-start_number", str(start_frame),
        "-i", str(COLOR_FRAMES_DIR / "frame_%03d.png"),
        "-frames:v", str(num_frames),
        "-c:v", "libx264", "-crf", "10", "-preset", "medium", "-pix_fmt", "yuv420p",
        str(ref_path)
    ]
    subprocess.run(cmd, capture_output=True)

    results = {}

    # 1. BASELINE at target size
    print(f"\n[1/3] Baseline at {target_size_mb} MB...")
    baseline_path, baseline_size, baseline_crf = encode_to_target_size(
        COLOR_FRAMES_DIR, work_dir / "baseline.mp4", start_frame, num_frames, target_size_mb
    )
    baseline_vmaf = compute_vmaf(ref_path, baseline_path)
    baseline_metrics = decode_and_measure(baseline_path, COLOR_FRAMES_DIR, start_frame, num_frames, work_dir / "baseline_work")
    print(f"    Size: {baseline_size:.2f} MB, CRF: {baseline_crf}, VMAF: {baseline_vmaf:.2f}, PSNR: {baseline_metrics['psnr']:.2f}")
    results['baseline'] = {'size': baseline_size, 'crf': baseline_crf, 'vmaf': baseline_vmaf, **baseline_metrics}

    # 2. BILATERAL at target size
    print(f"\n[2/3] Bilateral at {target_size_mb} MB...")
    bilateral_frames = preprocess_frames(start_frame, num_frames, work_dir / "bilateral_frames", "bilateral")
    bilateral_path, bilateral_size, bilateral_crf = encode_to_target_size(
        bilateral_frames, work_dir / "bilateral.mp4", start_frame, num_frames, target_size_mb
    )
    bilateral_vmaf = compute_vmaf(ref_path, bilateral_path)
    bilateral_metrics = decode_and_measure(bilateral_path, COLOR_FRAMES_DIR, start_frame, num_frames, work_dir / "bilateral_work")
    print(f"    Size: {bilateral_size:.2f} MB, CRF: {bilateral_crf}, VMAF: {bilateral_vmaf:.2f}, PSNR: {bilateral_metrics['psnr']:.2f}")
    results['bilateral'] = {'size': bilateral_size, 'crf': bilateral_crf, 'vmaf': bilateral_vmaf, **bilateral_metrics}

    # 3. BLUR at target size
    print(f"\n[3/3] Blur at {target_size_mb} MB...")
    blur_frames = preprocess_frames(start_frame, num_frames, work_dir / "blur_frames", "blur")
    blur_path, blur_size, blur_crf = encode_to_target_size(
        blur_frames, work_dir / "blur.mp4", start_frame, num_frames, target_size_mb
    )
    blur_vmaf = compute_vmaf(ref_path, blur_path)
    blur_metrics = decode_and_measure(blur_path, COLOR_FRAMES_DIR, start_frame, num_frames, work_dir / "blur_work")
    print(f"    Size: {blur_size:.2f} MB, CRF: {blur_crf}, VMAF: {blur_vmaf:.2f}, PSNR: {blur_metrics['psnr']:.2f}")
    results['blur'] = {'size': blur_size, 'crf': blur_crf, 'vmaf': blur_vmaf, **blur_metrics}

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS AT {target_size_mb} MB TARGET")
    print(f"{'='*70}")
    print(f"{'Method':<12} {'Size':>8} {'CRF':>6} {'VMAF':>8} {'PSNR':>8} {'ROI PSNR':>10}")
    print("-" * 60)
    for method, data in results.items():
        print(f"{method:<12} {data['size']:>7.2f}MB {data['crf']:>6} {data['vmaf']:>8.2f} {data['psnr']:>8.2f} {data['roi_psnr']:>10.2f}")

    # Create side-by-side
    print("\nCreating side-by-side comparison...")
    sbs_path = work_dir / "comparison.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(baseline_path), "-i", str(bilateral_path), "-i", str(blur_path),
        "-filter_complex",
        f"[0:v]scale=640:-1,drawtext=text='Baseline (CRF {baseline_crf})':fontsize=20:fontcolor=white:x=10:y=10[v0];"
        f"[1:v]scale=640:-1,drawtext=text='Bilateral (CRF {bilateral_crf})':fontsize=20:fontcolor=white:x=10:y=10[v1];"
        f"[2:v]scale=640:-1,drawtext=text='Blur (CRF {blur_crf})':fontsize=20:fontcolor=white:x=10:y=10[v2];"
        "[v0][v1][v2]hstack=inputs=3[out]",
        "-map", "[out]", "-c:v", "libx264", "-crf", "18", str(sbs_path)
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"Side-by-side: {sbs_path}")

    # Save results
    with open(work_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    # Test at different target sizes
    all_results = {}

    for target in [0.5, 1.0, 2.0, 3.0]:
        results = run_fair_comparison(target_size_mb=target, segment="rotation", num_frames=150)
        all_results[f"{target}mb"] = results

    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL TARGET SIZES")
    print("=" * 70)
    print(f"{'Target':<10} {'Baseline VMAF':>15} {'Bilateral VMAF':>15} {'Blur VMAF':>12}")
    print("-" * 55)

    for target, results in all_results.items():
        b = results['baseline']['vmaf']
        bi = results['bilateral']['vmaf']
        bl = results['blur']['vmaf']
        print(f"{target:<10} {b:>15.2f} {bi:>14.2f} ({bi-b:+.1f}) {bl:>11.2f} ({bl-b:+.1f})")
