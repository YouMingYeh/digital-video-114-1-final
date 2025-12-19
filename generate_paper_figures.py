"""
Generate publication-quality figures for the paper.
Reads experiment results and creates figures in report/ directory.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Style configuration for IEEE papers
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (3.5, 2.8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'baseline': '#2ecc71',  # Green
    'bilateral': '#3498db',  # Blue
    'blur': '#e74c3c',  # Red
}

MARKERS = {
    'baseline': 'o',
    'bilateral': 's',
    'blur': '^',
}


def load_results(results_dir: Path) -> dict:
    """Load experiment results from JSON files."""
    results = {}

    for segment in ['rotation', 'zoom']:
        results_file = results_dir / f'experiment_{segment}' / 'experiment_results.json'
        if results_file.exists():
            with open(results_file) as f:
                results[segment] = json.load(f)

    return results


def fig1_rd_curves(results: dict, output_dir: Path):
    """
    Figure 1: Rate-Distortion curves showing all methods.
    Two subplots: (a) Rotation segment, (b) Zoom segment
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    for idx, (segment, ax) in enumerate(zip(['rotation', 'zoom'], axes)):
        if segment not in results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'({chr(97+idx)}) {segment.capitalize()} Segment')
            continue

        data = results[segment]

        for method in ['baseline', 'bilateral', 'blur']:
            if method not in data['methods']:
                continue

            rd_points = data['methods'][method]['rd_points']
            bitrates = [p['bitrate_kbps'] for p in rd_points]
            psnrs = [p['psnr'] for p in rd_points]

            ax.plot(bitrates, psnrs,
                   marker=MARKERS[method],
                   color=COLORS[method],
                   label=method.capitalize(),
                   linewidth=1.5,
                   markersize=5)

        ax.set_xlabel('Bitrate (kbps)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'({chr(97+idx)}) {segment.capitalize()} Segment')
        ax.legend(loc='lower right')
        ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_rd_curves.png')
    plt.savefig(output_dir / 'fig1_rd_curves.pdf')
    plt.close()
    print(f"Generated: fig1_rd_curves.png")


def fig2_bd_rate(results: dict, output_dir: Path):
    """
    Figure 2: BD-Rate comparison bar chart.
    Shows BD-Rate for bilateral and blur vs baseline.
    """
    fig, ax = plt.subplots(figsize=(4, 3))

    segments = []
    bilateral_bd = []
    blur_bd = []

    for segment in ['rotation', 'zoom']:
        if segment not in results:
            continue

        data = results[segment]
        segments.append(segment.capitalize())

        bd_bilateral = data['methods'].get('bilateral', {}).get('bd_rate', 0) or 0
        bd_blur = data['methods'].get('blur', {}).get('bd_rate', 0) or 0

        bilateral_bd.append(bd_bilateral)
        blur_bd.append(bd_blur)

    if not segments:
        print("No BD-Rate data available")
        return

    x = np.arange(len(segments))
    width = 0.35

    bars1 = ax.bar(x - width/2, bilateral_bd, width, label='Bilateral', color=COLORS['bilateral'])
    bars2 = ax.bar(x + width/2, blur_bd, width, label='Blur', color=COLORS['blur'])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:+.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('BD-Rate (%)')
    ax.set_xlabel('Video Segment')
    ax.set_title('BD-Rate vs Baseline\n(positive = worse efficiency)')
    ax.set_xticks(x)
    ax.set_xticklabels(segments)
    ax.legend()

    # Color the region above 0 (bad) lightly
    ax.axhspan(0, ax.get_ylim()[1], alpha=0.1, color='red')
    ax.axhspan(ax.get_ylim()[0], 0, alpha=0.1, color='green')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_bd_rate.png')
    plt.savefig(output_dir / 'fig2_bd_rate.pdf')
    plt.close()
    print(f"Generated: fig2_bd_rate.png")


def fig3_quality_comparison(results: dict, output_dir: Path):
    """
    Figure 3: Quality comparison at CRF=28.
    Shows PSNR and SSIM for each method.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    # Get CRF=28 data point
    crf_target = 28

    for idx, (segment, ax) in enumerate(zip(['rotation', 'zoom'], axes)):
        if segment not in results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        data = results[segment]

        methods = []
        psnrs = []
        ssims = []

        for method in ['baseline', 'bilateral', 'blur']:
            if method not in data['methods']:
                continue

            rd_points = data['methods'][method]['rd_points']
            # Find CRF=28 or closest
            crf28_point = None
            for p in rd_points:
                if p['crf'] == crf_target:
                    crf28_point = p
                    break

            if crf28_point:
                methods.append(method.capitalize())
                psnrs.append(crf28_point['psnr'])
                ssims.append(crf28_point['ssim'])

        x = np.arange(len(methods))

        # PSNR bars
        bars = ax.bar(x, psnrs, color=[COLORS[m.lower()] for m in methods])

        # Add value labels
        for bar, psnr in zip(bars, psnrs):
            ax.annotate(f'{psnr:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, psnr),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'({chr(97+idx)}) {segment.capitalize()} @ CRF=28')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_quality_crf28.png')
    plt.savefig(output_dir / 'fig3_quality_crf28.pdf')
    plt.close()
    print(f"Generated: fig3_quality_crf28.png")


def fig4_size_vs_quality(results: dict, output_dir: Path):
    """
    Figure 4: Size reduction vs quality loss trade-off at CRF=28.
    Scatter plot showing the trade-off.
    """
    fig, ax = plt.subplots(figsize=(4, 3.5))

    crf_target = 28

    for segment in ['rotation', 'zoom']:
        if segment not in results:
            continue

        data = results[segment]

        # Get baseline reference
        baseline_point = None
        for p in data['methods'].get('baseline', {}).get('rd_points', []):
            if p['crf'] == crf_target:
                baseline_point = p
                break

        if not baseline_point:
            continue

        baseline_size = baseline_point['size_mb']
        baseline_psnr = baseline_point['psnr']

        for method in ['bilateral', 'blur']:
            if method not in data['methods']:
                continue

            for p in data['methods'][method]['rd_points']:
                if p['crf'] == crf_target:
                    size_reduction = (1 - p['size_mb'] / baseline_size) * 100
                    psnr_loss = baseline_psnr - p['psnr']

                    marker = 'o' if segment == 'rotation' else '^'
                    ax.scatter(size_reduction, psnr_loss,
                              color=COLORS[method],
                              marker=marker,
                              s=100,
                              label=f'{method.capitalize()} ({segment})')

    ax.set_xlabel('File Size Reduction (%)')
    ax.set_ylabel('PSNR Loss (dB)')
    ax.set_title('Trade-off: Size Reduction vs Quality Loss')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(loc='upper left', fontsize=8)

    # Add annotation
    ax.annotate('Better trade-off\n(lower & righter)',
               xy=(0.7, 0.1), xycoords='axes fraction',
               fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_tradeoff.png')
    plt.savefig(output_dir / 'fig4_tradeoff.pdf')
    plt.close()
    print(f"Generated: fig4_tradeoff.png")


def fig5_ssim_comparison(results: dict, output_dir: Path):
    """
    Figure 5: SSIM curves for structural quality comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    for idx, (segment, ax) in enumerate(zip(['rotation', 'zoom'], axes)):
        if segment not in results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'({chr(97+idx)}) {segment.capitalize()} Segment')
            continue

        data = results[segment]

        for method in ['baseline', 'bilateral', 'blur']:
            if method not in data['methods']:
                continue

            rd_points = data['methods'][method]['rd_points']
            bitrates = [p['bitrate_kbps'] for p in rd_points]
            ssims = [p['ssim'] for p in rd_points]

            ax.plot(bitrates, ssims,
                   marker=MARKERS[method],
                   color=COLORS[method],
                   label=method.capitalize(),
                   linewidth=1.5,
                   markersize=5)

        ax.set_xlabel('Bitrate (kbps)')
        ax.set_ylabel('SSIM')
        ax.set_title(f'({chr(97+idx)}) {segment.capitalize()} Segment')
        ax.legend(loc='lower right')
        ax.set_xlim(left=0)
        ax.set_ylim(0.9, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_ssim_curves.png')
    plt.savefig(output_dir / 'fig5_ssim_curves.pdf')
    plt.close()
    print(f"Generated: fig5_ssim_curves.png")


def generate_summary_table(results: dict, output_dir: Path):
    """Generate a summary table as text file."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT RESULTS SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    for segment in ['rotation', 'zoom']:
        if segment not in results:
            continue

        data = results[segment]
        lines.append(f"### {segment.upper()} SEGMENT ###")
        lines.append("")

        # BD-Rate
        lines.append("BD-Rate (vs Baseline):")
        for method in ['bilateral', 'blur']:
            bd = data['methods'].get(method, {}).get('bd_rate')
            if bd is not None:
                lines.append(f"  {method}: {bd:+.2f}%")
            else:
                lines.append(f"  {method}: N/A")
        lines.append("")

        # CRF=28 comparison
        lines.append("Quality at CRF=28:")
        lines.append(f"{'Method':<12} {'Bitrate':>10} {'PSNR':>8} {'SSIM':>8} {'Size':>8}")
        lines.append("-" * 50)

        for method in ['baseline', 'bilateral', 'blur']:
            if method not in data['methods']:
                continue

            for p in data['methods'][method]['rd_points']:
                if p['crf'] == 28:
                    lines.append(f"{method:<12} {p['bitrate_kbps']:>10.0f} {p['psnr']:>8.2f} {p['ssim']:>8.4f} {p['size_mb']:>7.2f}MB")

        lines.append("")
        lines.append("")

    # Write to file
    with open(output_dir / 'results_summary.txt', 'w') as f:
        f.write('\n'.join(lines))

    print(f"Generated: results_summary.txt")
    print('\n'.join(lines))


def main():
    output_dir = Path('outputs')
    report_dir = Path('report')
    report_dir.mkdir(exist_ok=True)

    print("Loading experiment results...")
    results = load_results(output_dir)

    if not results:
        print("ERROR: No experiment results found!")
        print("Run experiment_proper.py first.")
        return

    print(f"Found results for: {list(results.keys())}")
    print()

    # Generate all figures
    print("Generating figures...")
    fig1_rd_curves(results, report_dir)
    fig2_bd_rate(results, report_dir)
    fig3_quality_comparison(results, report_dir)
    fig4_size_vs_quality(results, report_dir)
    fig5_ssim_comparison(results, report_dir)

    print()
    generate_summary_table(results, report_dir)

    print()
    print("=" * 70)
    print("All figures generated in report/ directory")
    print("=" * 70)


if __name__ == "__main__":
    main()
