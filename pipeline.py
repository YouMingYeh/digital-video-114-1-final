"""
Depth-Aware Video Compression Pipeline - Experiment Runner

This module orchestrates the complete experimental workflow:
1. Encode videos (baseline vs depth-aware)
2. Decode back to frames
3. Compute depth-stratified quality metrics
4. Generate comparison report

Usage:
    python pipeline.py --num-frames 120 --crf 23
"""

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import (
    ExperimentConfig,
    EncoderConfig,
    DepthROIConfig,
    COLOR_FRAMES_DIR,
    DEPTH_FRAMES_DIR,
    OUTPUT_DIR,
    baseline_config,
    depth_aware_config,
    aggressive_depth_config,
)
from encoder import encode, decode_to_frames, get_video_info
from metrics import compute_video_metrics, VideoMetrics


def run_experiment(
    config: ExperimentConfig,
    compute_metrics: bool = True,
    cleanup_decoded: bool = True,
    sample_rate: int = 5  # Sample every Nth frame for faster metrics (1=all)
) -> dict:
    """
    Run a single encoding experiment.

    Returns a dictionary with all results including:
    - Encoding info (file size, bitrate)
    - Quality metrics (global and depth-stratified)
    """
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"  Codec: {config.encoder.codec}, CRF: {config.encoder.crf}")
    print(f"  Depth-aware: {config.use_depth_aware}")
    print(f"  Frames: {config.start_frame} to {config.start_frame + config.num_frames - 1}")
    print(f"{'='*60}")

    # Encode
    output_path = config.output_dir / f"{config.name}.mp4"
    encode_result = encode(config, output_path)

    print(f"  Encoded: {encode_result['file_size_mb']:.2f} MB")

    # Get video info
    video_info = get_video_info(output_path)
    print(f"  Bitrate: {video_info['bitrate_kbps']:.1f} kbps")

    result = {
        "config": {
            "name": config.name,
            "codec": config.encoder.codec,
            "crf": config.encoder.crf,
            "preset": config.encoder.preset,
            "use_depth_aware": config.use_depth_aware,
            "num_frames": config.num_frames,
        },
        "encoding": encode_result,
        "video_info": video_info,
    }

    # Compute quality metrics
    if compute_metrics:
        print("  Computing quality metrics...")

        # Decode to frames
        decoded_dir = config.output_dir / f"{config.name}_decoded"
        decoded_frames = decode_to_frames(output_path, decoded_dir)

        # Get reference and depth frame paths
        ref_frames = [
            COLOR_FRAMES_DIR / f"frame_{config.start_frame + i:03d}.png"
            for i in range(len(decoded_frames))
        ]
        depth_frames = [
            DEPTH_FRAMES_DIR / f"depth_{config.start_frame + i:03d}.png"
            for i in range(len(decoded_frames))
        ]

        # Compute metrics (sampled for speed)
        metrics = compute_video_metrics(
            ref_frames, decoded_frames, depth_frames,
            config.depth_roi,
            sample_rate=sample_rate
        )

        result["metrics"] = asdict(metrics)

        print(f"  PSNR Global: {metrics.psnr_global_mean:.2f} dB")
        print(f"  PSNR ROI (near): {metrics.psnr_roi_mean:.2f} dB")
        print(f"  PSNR Background (far): {metrics.psnr_background_mean:.2f} dB")
        print(f"  ROI Advantage: {metrics.roi_advantage_db:+.2f} dB")
        print(f"  SSIM: {metrics.ssim_global_mean:.4f}")

        # Cleanup decoded frames
        if cleanup_decoded:
            shutil.rmtree(decoded_dir, ignore_errors=True)

    return result


def run_comparison(
    num_frames: int = 120,
    crf: int = 23,
    output_json: Optional[Path] = None
) -> dict:
    """
    Run baseline vs depth-aware comparison experiment.
    """
    print("\n" + "=" * 70)
    print("DEPTH-AWARE VIDEO COMPRESSION COMPARISON")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "num_frames": num_frames,
            "crf": crf,
        },
        "experiments": []
    }

    # Run baseline
    baseline = baseline_config(num_frames=num_frames, crf=crf)
    baseline_result = run_experiment(baseline)
    results["experiments"].append(baseline_result)

    # Run depth-aware
    depth = depth_aware_config(num_frames=num_frames, crf=crf)
    depth_result = run_experiment(depth)
    results["experiments"].append(depth_result)

    # Run aggressive depth-aware
    aggressive = aggressive_depth_config(num_frames=num_frames, crf=crf)
    aggressive_result = run_experiment(aggressive)
    results["experiments"].append(aggressive_result)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    headers = ["Experiment", "Size (MB)", "Bitrate", "PSNR", "ROI PSNR", "BG PSNR", "ROI Adv"]
    print(f"{headers[0]:<25} {headers[1]:<10} {headers[2]:<10} {headers[3]:<8} {headers[4]:<10} {headers[5]:<10} {headers[6]:<8}")
    print("-" * 90)

    for exp in results["experiments"]:
        name = exp["config"]["name"]
        size = exp["encoding"]["file_size_mb"]
        bitrate = exp["video_info"]["bitrate_kbps"]
        metrics = exp.get("metrics", {})

        psnr = metrics.get("psnr_global_mean", 0)
        roi_psnr = metrics.get("psnr_roi_mean", 0)
        bg_psnr = metrics.get("psnr_background_mean", 0)
        roi_adv = roi_psnr - bg_psnr

        print(f"{name:<25} {size:<10.2f} {bitrate:<10.1f} {psnr:<8.2f} {roi_psnr:<10.2f} {bg_psnr:<10.2f} {roi_adv:<+8.2f}")

    # Save results
    if output_json is None:
        output_json = OUTPUT_DIR / "comparison_results.json"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_json}")

    return results


def run_bitrate_matched_comparison(
    num_frames: int = 60,
    target_crf: int = 23,
    output_json: Optional[Path] = None
) -> dict:
    """
    Fair comparison: match bitrates, then compare ROI quality.

    Since depth-aware with negative ROI qoffset increases bitrate,
    we compensate by using higher CRF for depth-aware encoding.
    This shows the true benefit: better ROI quality at same bitrate.
    """
    print("\n" + "=" * 70)
    print("BITRATE-MATCHED COMPARISON")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "type": "bitrate_matched",
        "experiments": []
    }

    # Run baseline
    baseline = baseline_config(num_frames=num_frames, crf=target_crf)
    baseline_result = run_experiment(baseline)
    results["experiments"].append(baseline_result)
    baseline_bitrate = baseline_result["video_info"]["bitrate_kbps"]

    # Run depth-aware at higher CRF to match bitrate
    # Depth-aware typically uses ~2x more bits, so increase CRF by ~6
    # (each +6 CRF roughly halves bitrate)
    depth_crf = target_crf + 6
    depth = ExperimentConfig(
        name=f"depth_aware_crf{depth_crf}_matched",
        num_frames=num_frames,
        encoder=EncoderConfig(crf=depth_crf, aq_mode=1),
        depth_roi=DepthROIConfig(
            depth_mode="near_high",
            near_qoffset=-0.3,
            far_qoffset=0.3,
        ),
        use_depth_aware=True,
    )
    depth_result = run_experiment(depth)
    results["experiments"].append(depth_result)

    # Calculate improvement
    baseline_roi = baseline_result["metrics"]["psnr_roi_mean"]
    depth_roi = depth_result["metrics"]["psnr_roi_mean"]
    depth_bitrate = depth_result["video_info"]["bitrate_kbps"]

    roi_improvement = depth_roi - baseline_roi
    bitrate_ratio = depth_bitrate / baseline_bitrate

    print("\n" + "=" * 70)
    print("BITRATE-MATCHED RESULTS")
    print("=" * 70)
    print(f"Baseline (CRF {target_crf}):")
    print(f"  Bitrate: {baseline_bitrate:.1f} kbps")
    print(f"  ROI PSNR: {baseline_roi:.2f} dB")
    print(f"\nDepth-Aware (CRF {depth_crf}, matched):")
    print(f"  Bitrate: {depth_bitrate:.1f} kbps ({bitrate_ratio:.1%} of baseline)")
    print(f"  ROI PSNR: {depth_roi:.2f} dB")
    print(f"\n>>> ROI PSNR IMPROVEMENT: {roi_improvement:+.2f} dB <<<")

    results["summary"] = {
        "baseline_crf": target_crf,
        "depth_crf": depth_crf,
        "baseline_bitrate_kbps": baseline_bitrate,
        "depth_bitrate_kbps": depth_bitrate,
        "bitrate_ratio": bitrate_ratio,
        "baseline_roi_psnr": baseline_roi,
        "depth_roi_psnr": depth_roi,
        "roi_psnr_improvement_db": roi_improvement,
    }

    if output_json is None:
        output_json = OUTPUT_DIR / "bitrate_matched_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_json}")
    return results


def run_crf_sweep(
    num_frames: int = 60,
    crf_values: list[int] = [18, 23, 28, 33],
    output_json: Optional[Path] = None
) -> dict:
    """
    Run experiments across multiple CRF values to generate rate-distortion data.
    """
    print("\n" + "=" * 70)
    print("CRF SWEEP EXPERIMENT")
    print(f"CRF values: {crf_values}")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "num_frames": num_frames,
            "crf_values": crf_values,
        },
        "experiments": []
    }

    for crf in crf_values:
        print(f"\n>>> CRF = {crf}")

        # Baseline
        baseline = baseline_config(num_frames=num_frames, crf=crf)
        baseline_result = run_experiment(baseline)
        results["experiments"].append(baseline_result)

        # Depth-aware
        depth = depth_aware_config(num_frames=num_frames, crf=crf)
        depth_result = run_experiment(depth)
        results["experiments"].append(depth_result)

    # Save results
    if output_json is None:
        output_json = OUTPUT_DIR / "crf_sweep_results.json"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Depth-Aware Video Compression Pipeline"
    )
    parser.add_argument(
        "--num-frames", "-n",
        type=int,
        default=120,
        help="Number of frames to encode (default: 120)"
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="CRF value for encoding (default: 23)"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run CRF sweep instead of single comparison"
    )
    parser.add_argument(
        "--matched",
        action="store_true",
        help="Run bitrate-matched comparison (fair comparison)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSON path"
    )

    args = parser.parse_args()

    if args.matched:
        run_bitrate_matched_comparison(
            num_frames=args.num_frames,
            target_crf=args.crf,
            output_json=args.output
        )
    elif args.sweep:
        run_crf_sweep(
            num_frames=args.num_frames,
            output_json=args.output
        )
    else:
        run_comparison(
            num_frames=args.num_frames,
            crf=args.crf,
            output_json=args.output
        )


if __name__ == "__main__":
    main()
