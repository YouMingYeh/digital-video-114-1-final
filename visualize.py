"""
Depth-Aware Video Compression Pipeline - Visualization

This module generates charts and visual reports from experiment results.

Charts:
-------
1. Rate-Distortion curves (bitrate vs PSNR)
2. ROI vs Background quality comparison
3. Depth map and importance map visualization
4. Per-frame quality plots
"""

import json
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

# Optional matplotlib import with fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization features disabled.")

from config import OUTPUT_DIR, COLOR_FRAMES_DIR, DEPTH_FRAMES_DIR, DepthROIConfig
from depth import load_depth_map, create_importance_map, depth_to_roi_regions


def ensure_matplotlib():
    """Check matplotlib availability."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization. Install with: pip install matplotlib")


def plot_rd_curve(results_json: Path, output_path: Optional[Path] = None):
    """
    Plot Rate-Distortion curve from CRF sweep results.

    Shows bitrate (x-axis) vs PSNR (y-axis) for baseline and depth-aware.
    """
    ensure_matplotlib()

    with open(results_json) as f:
        data = json.load(f)

    # Separate experiments by method
    baseline_data = []
    depth_data = []

    for exp in data["experiments"]:
        bitrate = exp["video_info"]["bitrate_kbps"]
        psnr = exp["metrics"]["psnr_global_mean"]
        name = exp["config"]["name"]

        if "baseline" in name:
            baseline_data.append((bitrate, psnr))
        else:
            depth_data.append((bitrate, psnr))

    # Sort by bitrate
    baseline_data.sort(key=lambda x: x[0])
    depth_data.sort(key=lambda x: x[0])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    if baseline_data:
        br, psnr = zip(*baseline_data)
        ax.plot(br, psnr, 'o-', label='Baseline', color='#2196F3', linewidth=2, markersize=8)

    if depth_data:
        br, psnr = zip(*depth_data)
        ax.plot(br, psnr, 's-', label='Depth-Aware', color='#4CAF50', linewidth=2, markersize=8)

    ax.set_xlabel('Bitrate (kbps)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Rate-Distortion Curve: Baseline vs Depth-Aware Encoding', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        output_path = OUTPUT_DIR / "rd_curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved RD curve to: {output_path}")


def plot_roi_comparison(results_json: Path, output_path: Optional[Path] = None):
    """
    Bar chart comparing ROI vs Background PSNR across experiments.
    """
    ensure_matplotlib()

    with open(results_json) as f:
        data = json.load(f)

    experiments = []
    for exp in data["experiments"]:
        experiments.append({
            "name": exp["config"]["name"].replace("_", "\n"),
            "roi_psnr": exp["metrics"]["psnr_roi_mean"],
            "bg_psnr": exp["metrics"]["psnr_background_mean"],
            "global_psnr": exp["metrics"]["psnr_global_mean"],
        })

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(experiments))
    width = 0.25

    roi_vals = [e["roi_psnr"] for e in experiments]
    bg_vals = [e["bg_psnr"] for e in experiments]
    global_vals = [e["global_psnr"] for e in experiments]

    bars1 = ax.bar(x - width, roi_vals, width, label='ROI (Near)', color='#4CAF50')
    bars2 = ax.bar(x, bg_vals, width, label='Background (Far)', color='#9E9E9E')
    bars3 = ax.bar(x + width, global_vals, width, label='Global', color='#2196F3')

    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Quality Comparison: ROI vs Background', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([e["name"] for e in experiments], fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if output_path is None:
        output_path = OUTPUT_DIR / "roi_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved ROI comparison to: {output_path}")


def visualize_depth_analysis(
    frame_index: int = 0,
    config: Optional[DepthROIConfig] = None,
    output_path: Optional[Path] = None
):
    """
    Visualize depth map processing for a single frame.

    Shows: Original → Depth Map → Importance Map → ROI Overlay
    """
    ensure_matplotlib()

    if config is None:
        config = DepthROIConfig()

    color_path = COLOR_FRAMES_DIR / f"frame_{frame_index:03d}.png"
    depth_path = DEPTH_FRAMES_DIR / f"depth_{frame_index:03d}.png"

    if not color_path.exists() or not depth_path.exists():
        raise FileNotFoundError(f"Frame {frame_index} not found")

    # Load data
    color_img = np.array(Image.open(color_path))
    depth = load_depth_map(depth_path)
    importance = create_importance_map(depth, config)
    regions = depth_to_roi_regions(depth_path, config, grid_size=8)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Original frame
    axes[0, 0].imshow(color_img)
    axes[0, 0].set_title('Original Frame', fontsize=12)
    axes[0, 0].axis('off')

    # Depth map
    im1 = axes[0, 1].imshow(depth, cmap='viridis')
    axes[0, 1].set_title('Depth Map', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Importance map
    im2 = axes[1, 0].imshow(importance, cmap='RdYlGn')
    axes[1, 0].set_title('Importance Map (Green=Near, Red=Far)', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # ROI overlay
    axes[1, 1].imshow(color_img)
    h, w = color_img.shape[:2]

    # Draw ROI regions
    for region in regions:
        x = region.x * w
        y = region.y * h
        rw = region.w * w
        rh = region.h * h

        # Color based on qoffset: green = better quality, red = worse quality
        if region.qoffset < 0:
            color = (0, 1, 0, 0.3)  # Green for high quality
            edge_color = 'green'
        else:
            color = (1, 0, 0, 0.3)  # Red for low quality
            edge_color = 'red'

        rect = mpatches.Rectangle(
            (x, y), rw, rh,
            linewidth=1,
            edgecolor=edge_color,
            facecolor=color
        )
        axes[1, 1].add_patch(rect)

    axes[1, 1].set_title(f'ROI Regions ({len(regions)} regions)', fontsize=12)
    axes[1, 1].axis('off')

    # Add legend
    green_patch = mpatches.Patch(color='green', alpha=0.5, label='High Quality (Near)')
    red_patch = mpatches.Patch(color='red', alpha=0.5, label='Low Quality (Far)')
    axes[1, 1].legend(handles=[green_patch, red_patch], loc='upper right')

    plt.suptitle(f'Depth-Aware Encoding Analysis - Frame {frame_index}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path is None:
        output_path = OUTPUT_DIR / f"depth_analysis_frame{frame_index:03d}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved depth analysis to: {output_path}")


def generate_report(results_json: Path, output_dir: Optional[Path] = None):
    """
    Generate complete visual report from experiment results.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating visual report...")

    # Generate all charts
    try:
        plot_rd_curve(results_json, output_dir / "rd_curve.png")
    except Exception as e:
        print(f"  Warning: Could not generate RD curve: {e}")

    try:
        plot_roi_comparison(results_json, output_dir / "roi_comparison.png")
    except Exception as e:
        print(f"  Warning: Could not generate ROI comparison: {e}")

    try:
        visualize_depth_analysis(0, output_path=output_dir / "depth_analysis.png")
    except Exception as e:
        print(f"  Warning: Could not generate depth analysis: {e}")

    print(f"\nReport saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument(
        "--results", "-r",
        type=Path,
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--depth-frame", "-d",
        type=int,
        default=0,
        help="Frame index for depth analysis visualization"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for visualizations"
    )

    args = parser.parse_args()

    if args.results:
        generate_report(args.results, args.output)
    else:
        # Just generate depth analysis
        visualize_depth_analysis(args.depth_frame, output_path=args.output)
