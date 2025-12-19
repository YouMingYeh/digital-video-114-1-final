"""
Quick Benchmark for Depth-Guided Video Compression

Compares:
1. Baseline: Standard RGB video encoding
2. Bilateral: Depth-guided bilateral preprocessing
3. Blur: Depth-guided Gaussian blur preprocessing

For rigorous evaluation with BD-Rate, use experiment_proper.py instead.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict

from config import OUTPUT_DIR, COLOR_FRAMES_DIR, EncoderConfig, DepthConfig
from utils import encode_frames_to_video, get_video_size_mb
from method2_depth_quality import encode_depth_quality


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    method: str
    segment: str
    crf: int
    size_mb: float
    psnr: float = 0.0
    ssim: float = 0.0


def encode_baseline(start_frame: int, num_frames: int, output_dir: Path, encoder_config: EncoderConfig) -> dict:
    """Encode baseline (standard RGB video)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline.mp4"

    encode_frames_to_video(
        COLOR_FRAMES_DIR, output_path,
        start_number=start_frame,
        num_frames=num_frames,
        encoder_args=encoder_config.to_ffmpeg_args()
    )

    return {"output_path": output_path, "size_mb": get_video_size_mb(output_path)}


def run_benchmark(segment: str = "rotation", crf: int = 28, num_frames: int = 100) -> list[BenchmarkResult]:
    """
    Run quick benchmark comparing baseline vs preprocessing methods.

    For rigorous BD-Rate evaluation, use experiment_proper.py instead.
    """
    start_frame = 0 if segment == "rotation" else 300
    output_dir = OUTPUT_DIR / f"benchmark_{segment}_crf{crf}"
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_config = EncoderConfig(crf=crf, preset="medium")
    depth_config = DepthConfig()
    results = []

    print("=" * 60)
    print(f"BENCHMARK: {segment.upper()}, CRF={crf}, {num_frames} frames")
    print("=" * 60)

    # 1. Baseline
    print("\n[1/3] Baseline...")
    baseline = encode_baseline(start_frame, num_frames, output_dir / "baseline", encoder_config)
    print(f"      Size: {baseline['size_mb']:.2f} MB")
    results.append(BenchmarkResult("baseline", segment, crf, baseline["size_mb"]))

    # 2. Bilateral
    print("\n[2/3] Bilateral preprocessing...")
    try:
        bilateral = encode_depth_quality(start_frame, num_frames, output_dir / "bilateral",
                                         encoder_config, depth_config, method="bilateral")
        print(f"      Size: {bilateral['size_mb']:.2f} MB")
        results.append(BenchmarkResult("bilateral", segment, crf, bilateral["size_mb"]))
    except Exception as e:
        print(f"      Error: {e}")

    # 3. Blur
    print("\n[3/3] Blur preprocessing...")
    try:
        blur = encode_depth_quality(start_frame, num_frames, output_dir / "blur",
                                    encoder_config, depth_config, method="blur")
        print(f"      Size: {blur['size_mb']:.2f} MB")
        results.append(BenchmarkResult("blur", segment, crf, blur["size_mb"]))
    except Exception as e:
        print(f"      Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (size only - use experiment_proper.py for BD-Rate)")
    print("=" * 60)
    print(f"{'Method':<15} {'Size (MB)':>10} {'vs Baseline':>15}")
    print("-" * 45)

    baseline_size = results[0].size_mb
    for r in results:
        diff = ((r.size_mb / baseline_size) - 1) * 100 if baseline_size > 0 else 0
        diff_str = f"{diff:+.1f}%" if r.method != "baseline" else "--"
        print(f"{r.method:<15} {r.size_mb:>10.2f} {diff_str:>15}")

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    return results


if __name__ == "__main__":
    run_benchmark(segment="rotation", crf=28, num_frames=100)
