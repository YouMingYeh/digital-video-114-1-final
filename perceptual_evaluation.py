"""
Perceptual Evaluation for Depth-Guided Video Compression

Key insight: Humans focus on foreground, not background.
PSNR treats all pixels equally - but that's not how humans perceive video.

This script evaluates:
1. VMAF - Netflix's perceptual quality metric
2. Weighted PSNR - Foreground pixels count more
3. ROI-only metrics - Quality where humans actually look
4. Side-by-side comparisons for subjective evaluation
"""

import json
import subprocess
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from PIL import Image
import tempfile

from config import OUTPUT_DIR, COLOR_FRAMES_DIR, DEPTH_FRAMES_DIR, DepthConfig
from utils import load_color_frame, load_depth_frame, depth_to_importance, compute_psnr, compute_ssim
from method2_depth_quality import apply_depth_quality_bilateral, apply_depth_quality_blur


@dataclass
class PerceptualResult:
    """Perceptual quality metrics."""
    method: str
    segment: str
    size_mb: float
    # Standard metrics (computer perspective)
    psnr: float
    ssim: float
    # Perceptual metrics (human perspective)
    vmaf: float
    weighted_psnr: float  # Foreground weighted 3x
    roi_psnr: float       # Foreground only
    roi_ssim: float


def compute_vmaf(reference_video: Path, distorted_video: Path) -> float:
    """
    Compute VMAF score using FFmpeg.

    VMAF (Video Multi-method Assessment Fusion) is Netflix's perceptual
    quality metric that correlates well with human perception.

    Score range: 0-100 (higher = better perceptual quality)
    """
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


def compute_weighted_psnr(
    original: np.ndarray,
    decoded: np.ndarray,
    importance: np.ndarray,
    foreground_weight: float = 3.0
) -> float:
    """
    Compute weighted PSNR where foreground pixels count more.

    This better reflects human perception - we care more about
    foreground quality than background quality.
    """
    # Create weight map: foreground gets higher weight
    weights = 1.0 + (foreground_weight - 1.0) * importance
    weights = weights / weights.mean()  # Normalize

    # Expand weights to match RGB channels
    if original.ndim == 3:
        weights = np.expand_dims(weights, axis=-1)

    # Weighted MSE
    diff = (original.astype(float) - decoded.astype(float)) ** 2
    weighted_mse = np.sum(weights * diff) / np.sum(weights * np.ones_like(diff))

    if weighted_mse == 0:
        return float('inf')

    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(weighted_mse))


def encode_at_target_size(
    frames_dir: Path,
    target_size_mb: float,
    output_path: Path,
    start_frame: int,
    num_frames: int,
    tolerance: float = 0.1
) -> tuple[Path, float]:
    """
    Encode video to approximately match target file size.
    Uses binary search on CRF to find the right quality level.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Binary search for CRF that gives target size
    crf_low, crf_high = 15, 45
    best_crf = 28
    best_size = 0

    for _ in range(8):  # Max 8 iterations
        crf = (crf_low + crf_high) // 2

        cmd = [
            "ffmpeg", "-y",
            "-framerate", "30",
            "-start_number", str(start_frame),
            "-i", str(frames_dir / "frame_%03d.png"),
            "-frames:v", str(num_frames),
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)

        size_mb = output_path.stat().st_size / (1024 * 1024)

        if abs(size_mb - target_size_mb) / target_size_mb < tolerance:
            return output_path, size_mb

        if size_mb > target_size_mb:
            crf_low = crf
        else:
            crf_high = crf

        if abs(size_mb - target_size_mb) < abs(best_size - target_size_mb):
            best_crf = crf
            best_size = size_mb

    return output_path, best_size


def create_reference_video(start_frame: int, num_frames: int, output_path: Path) -> Path:
    """Create high-quality reference video for VMAF comparison."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-start_number", str(start_frame),
        "-i", str(COLOR_FRAMES_DIR / "frame_%03d.png"),
        "-frames:v", str(num_frames),
        "-c:v", "libx264",
        "-crf", "10",  # Very high quality reference
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def preprocess_frames_to_dir(
    start_frame: int,
    num_frames: int,
    output_dir: Path,
    method: str
) -> Path:
    """Preprocess frames and save to directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_config = DepthConfig()

    for i in range(num_frames):
        frame_idx = start_frame + i
        color = load_color_frame(frame_idx)
        depth = load_depth_frame(frame_idx)

        if method == "bilateral":
            processed = apply_depth_quality_bilateral(color, depth, depth_config)
        else:
            processed = apply_depth_quality_blur(color, depth, depth_config)

        Image.fromarray(processed).save(output_dir / f"frame_{frame_idx:03d}.png")

    return output_dir


def compute_frame_metrics(
    original_dir: Path,
    decoded_dir: Path,
    start_frame: int,
    num_frames: int
) -> dict:
    """Compute per-frame quality metrics with importance weighting."""
    psnr_list, ssim_list = [], []
    weighted_psnr_list = []
    roi_psnr_list, roi_ssim_list = [], []

    for i in range(num_frames):
        orig_idx = start_frame + i
        dec_idx = i + 1  # Decoded frames start at 1

        orig_path = original_dir / f"frame_{orig_idx:03d}.png"
        dec_path = decoded_dir / f"frame_{dec_idx:03d}.png"

        if not orig_path.exists() or not dec_path.exists():
            continue

        orig = np.array(Image.open(orig_path).convert("RGB"))
        dec = np.array(Image.open(dec_path).convert("RGB"))
        depth = load_depth_frame(orig_idx)
        importance = depth_to_importance(depth, "near_high")

        # Standard metrics
        psnr_list.append(compute_psnr(orig, dec))
        ssim_list.append(compute_ssim(orig, dec))

        # Weighted PSNR (foreground 3x weight)
        weighted_psnr_list.append(compute_weighted_psnr(orig, dec, importance, 3.0))

        # ROI-only metrics
        roi_mask = importance > 0.5
        if roi_mask.any():
            roi_psnr_list.append(compute_psnr(orig[roi_mask], dec[roi_mask]))
            # For SSIM, create masked images
            roi_orig = orig.copy()
            roi_dec = dec.copy()
            roi_orig[~roi_mask] = 0
            roi_dec[~roi_mask] = 0
            roi_ssim_list.append(compute_ssim(roi_orig, roi_dec))

    return {
        'psnr': np.mean(psnr_list) if psnr_list else 0,
        'ssim': np.mean(ssim_list) if ssim_list else 0,
        'weighted_psnr': np.mean(weighted_psnr_list) if weighted_psnr_list else 0,
        'roi_psnr': np.mean(roi_psnr_list) if roi_psnr_list else 0,
        'roi_ssim': np.mean(roi_ssim_list) if roi_ssim_list else 0,
    }


def decode_video(video_path: Path, output_dir: Path) -> Path:
    """Decode video to frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        str(output_dir / "frame_%03d.png")
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_dir


def run_perceptual_comparison(
    segment: str = "rotation",
    num_frames: int = 150,
    target_size_mb: float = None
) -> dict:
    """
    Run perceptual quality comparison at SAME file size.

    This is the fair comparison humans care about:
    "Given the same bandwidth, which method looks better?"
    """
    start_frame = 0 if segment == "rotation" else 300
    work_dir = OUTPUT_DIR / f"perceptual_{segment}"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"PERCEPTUAL EVALUATION: {segment.upper()} segment")
    print("=" * 70)

    # Step 1: Create reference video for VMAF
    print("\n[1/6] Creating reference video...")
    ref_video = create_reference_video(start_frame, num_frames, work_dir / "reference.mp4")

    # Step 2: Encode baseline to get target size
    print("\n[2/6] Encoding baseline...")
    baseline_dir = work_dir / "baseline"
    baseline_dir.mkdir(exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-start_number", str(start_frame),
        "-i", str(COLOR_FRAMES_DIR / "frame_%03d.png"),
        "-frames:v", str(num_frames),
        "-c:v", "libx264", "-crf", "28", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        str(baseline_dir / "video.mp4")
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    baseline_video = baseline_dir / "video.mp4"
    baseline_size = baseline_video.stat().st_size / (1024 * 1024)

    if target_size_mb is None:
        target_size_mb = baseline_size

    print(f"    Baseline size: {baseline_size:.2f} MB (target for others)")

    # Step 3: Preprocess frames
    print("\n[3/6] Preprocessing frames...")
    bilateral_frames = preprocess_frames_to_dir(
        start_frame, num_frames, work_dir / "bilateral_frames", "bilateral"
    )
    blur_frames = preprocess_frames_to_dir(
        start_frame, num_frames, work_dir / "blur_frames", "blur"
    )

    # Step 4: Encode preprocessed at SAME target size
    print("\n[4/6] Encoding preprocessed videos at same size...")

    bilateral_video, bilateral_size = encode_at_target_size(
        bilateral_frames, target_size_mb,
        work_dir / "bilateral" / "video.mp4",
        start_frame, num_frames
    )
    print(f"    Bilateral: {bilateral_size:.2f} MB")

    blur_video, blur_size = encode_at_target_size(
        blur_frames, target_size_mb,
        work_dir / "blur" / "video.mp4",
        start_frame, num_frames
    )
    print(f"    Blur: {blur_size:.2f} MB")

    # Step 5: Compute VMAF
    print("\n[5/6] Computing VMAF (perceptual quality)...")
    vmaf_baseline = compute_vmaf(ref_video, baseline_video)
    vmaf_bilateral = compute_vmaf(ref_video, bilateral_video)
    vmaf_blur = compute_vmaf(ref_video, blur_video)

    print(f"    Baseline VMAF:   {vmaf_baseline:.2f}")
    print(f"    Bilateral VMAF:  {vmaf_bilateral:.2f} ({vmaf_bilateral - vmaf_baseline:+.2f})")
    print(f"    Blur VMAF:       {vmaf_blur:.2f} ({vmaf_blur - vmaf_baseline:+.2f})")

    # Step 6: Compute frame-level metrics
    print("\n[6/6] Computing detailed metrics...")

    # Decode videos
    baseline_decoded = decode_video(baseline_video, work_dir / "baseline_decoded")
    bilateral_decoded = decode_video(bilateral_video, work_dir / "bilateral_decoded")
    blur_decoded = decode_video(blur_video, work_dir / "blur_decoded")

    # Compute metrics
    baseline_metrics = compute_frame_metrics(COLOR_FRAMES_DIR, baseline_decoded, start_frame, num_frames)
    bilateral_metrics = compute_frame_metrics(COLOR_FRAMES_DIR, bilateral_decoded, start_frame, num_frames)
    blur_metrics = compute_frame_metrics(COLOR_FRAMES_DIR, blur_decoded, start_frame, num_frames)

    # Create results
    results = {
        "segment": segment,
        "num_frames": num_frames,
        "target_size_mb": target_size_mb,
        "methods": {
            "baseline": PerceptualResult(
                method="baseline",
                segment=segment,
                size_mb=baseline_size,
                psnr=baseline_metrics['psnr'],
                ssim=baseline_metrics['ssim'],
                vmaf=vmaf_baseline,
                weighted_psnr=baseline_metrics['weighted_psnr'],
                roi_psnr=baseline_metrics['roi_psnr'],
                roi_ssim=baseline_metrics['roi_ssim']
            ),
            "bilateral": PerceptualResult(
                method="bilateral",
                segment=segment,
                size_mb=bilateral_size,
                psnr=bilateral_metrics['psnr'],
                ssim=bilateral_metrics['ssim'],
                vmaf=vmaf_bilateral,
                weighted_psnr=bilateral_metrics['weighted_psnr'],
                roi_psnr=bilateral_metrics['roi_psnr'],
                roi_ssim=bilateral_metrics['roi_ssim']
            ),
            "blur": PerceptualResult(
                method="blur",
                segment=segment,
                size_mb=blur_size,
                psnr=blur_metrics['psnr'],
                ssim=blur_metrics['ssim'],
                vmaf=vmaf_blur,
                weighted_psnr=blur_metrics['weighted_psnr'],
                roi_psnr=blur_metrics['roi_psnr'],
                roi_ssim=blur_metrics['roi_ssim']
            )
        }
    }

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS AT SAME FILE SIZE")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Baseline':>12} {'Bilateral':>12} {'Blur':>12}")
    print("-" * 60)
    print(f"{'Size (MB)':<20} {baseline_size:>12.2f} {bilateral_size:>12.2f} {blur_size:>12.2f}")
    print(f"{'PSNR (dB)':<20} {baseline_metrics['psnr']:>12.2f} {bilateral_metrics['psnr']:>12.2f} {blur_metrics['psnr']:>12.2f}")
    print(f"{'VMAF':<20} {vmaf_baseline:>12.2f} {vmaf_bilateral:>12.2f} {vmaf_blur:>12.2f}")
    print(f"{'Weighted PSNR':<20} {baseline_metrics['weighted_psnr']:>12.2f} {bilateral_metrics['weighted_psnr']:>12.2f} {blur_metrics['weighted_psnr']:>12.2f}")
    print(f"{'ROI PSNR':<20} {baseline_metrics['roi_psnr']:>12.2f} {bilateral_metrics['roi_psnr']:>12.2f} {blur_metrics['roi_psnr']:>12.2f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
At the SAME file size:
- PSNR: Standard metric (all pixels equal) - favors baseline
- VMAF: Perceptual metric (models human vision) - may favor preprocessing
- Weighted PSNR: Foreground counts 3x more - tests our hypothesis
- ROI PSNR: Foreground only - what humans actually look at

If VMAF or Weighted PSNR favors preprocessing, our method has merit
for PERCEPTUAL quality even if standard PSNR is worse.
""")

    # Save results
    results_file = work_dir / "perceptual_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "segment": segment,
            "target_size_mb": target_size_mb,
            "methods": {k: asdict(v) for k, v in results["methods"].items()}
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results


def create_side_by_side_video(segment: str = "rotation", num_frames: int = 150):
    """Create side-by-side comparison video for subjective evaluation."""
    start_frame = 0 if segment == "rotation" else 300
    work_dir = OUTPUT_DIR / f"perceptual_{segment}"

    print("\nCreating side-by-side comparison video...")

    # Check if videos exist
    baseline_video = work_dir / "baseline" / "video.mp4"
    bilateral_video = work_dir / "bilateral" / "video.mp4"
    blur_video = work_dir / "blur" / "video.mp4"

    if not all(v.exists() for v in [baseline_video, bilateral_video, blur_video]):
        print("Run perceptual comparison first!")
        return

    output_video = work_dir / "comparison_side_by_side.mp4"

    # Create 3-way side-by-side with labels
    cmd = [
        "ffmpeg", "-y",
        "-i", str(baseline_video),
        "-i", str(bilateral_video),
        "-i", str(blur_video),
        "-filter_complex",
        "[0:v]scale=640:-1,drawtext=text='Baseline':fontsize=24:fontcolor=white:x=10:y=10[v0];"
        "[1:v]scale=640:-1,drawtext=text='Bilateral':fontsize=24:fontcolor=white:x=10:y=10[v1];"
        "[2:v]scale=640:-1,drawtext=text='Blur':fontsize=24:fontcolor=white:x=10:y=10[v2];"
        "[v0][v1][v2]hstack=inputs=3[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "18",
        str(output_video)
    ]

    subprocess.run(cmd, capture_output=True)
    print(f"Side-by-side video saved to: {output_video}")

    return output_video


if __name__ == "__main__":
    # Run perceptual evaluation on rotation segment
    results = run_perceptual_comparison(segment="rotation", num_frames=150)

    # Create side-by-side comparison
    create_side_by_side_video(segment="rotation")

    # Also run on zoom segment
    results_zoom = run_perceptual_comparison(segment="zoom", num_frames=150)
    create_side_by_side_video(segment="zoom")
