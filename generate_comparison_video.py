"""
Generate side-by-side video comparisons for the paper.
"""

import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    ExperimentConfig, EncoderConfig, DepthROIConfig,
    COLOR_FRAMES_DIR, OUTPUT_DIR
)
from pipeline import run_experiment
from encoder import decode_to_frames

COMPARISON_DIR = OUTPUT_DIR / "comparisons"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def create_side_by_side_frame(ref_path, base_path, depth_path, output_path, crf=40):
    """Create a single side-by-side comparison frame with labels."""
    ref = Image.open(ref_path)
    base = Image.open(base_path)
    depth = Image.open(depth_path)

    w, h = ref.size

    # Create canvas for 3 images side by side
    canvas = Image.new('RGB', (w * 3, h + 60), color=(30, 30, 30))

    # Paste images
    canvas.paste(ref, (0, 60))
    canvas.paste(base, (w, 60))
    canvas.paste(depth, (w * 2, 60))

    # Add labels
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()
        small_font = font

    # Labels
    draw.text((w//2 - 80, 15), "Reference", fill=(255, 255, 255), font=font)
    draw.text((w + w//2 - 120, 15), f"Baseline CRF {crf}", fill=(100, 180, 255), font=font)
    draw.text((w*2 + w//2 - 160, 15), f"Depth-Aware CRF {crf}", fill=(100, 255, 150), font=font)

    canvas.save(output_path)
    return canvas


def create_comparison_video(num_frames=500, crf=40):
    """Generate full comparison video: Reference | Baseline | Depth-Aware."""
    print(f"\n{'='*60}")
    print(f"Generating comparison video ({num_frames} frames, CRF {crf})")
    print(f"{'='*60}")

    # Run baseline encoding
    base_config = ExperimentConfig(
        name=f'comparison_base_crf{crf}',
        num_frames=num_frames,
        encoder=EncoderConfig(crf=crf, preset='ultrafast'),
        use_depth_aware=False,
    )
    base_result = run_experiment(base_config, compute_metrics=True, cleanup_decoded=False, sample_rate=10)

    # Run depth-aware encoding
    depth_config = ExperimentConfig(
        name=f'comparison_depth_crf{crf}',
        num_frames=num_frames,
        encoder=EncoderConfig(crf=crf, preset='ultrafast'),
        depth_roi=DepthROIConfig(
            depth_mode='near_high',
            num_quality_levels=5,
            near_qoffset=-0.4,
            far_qoffset=0.4,
        ),
        use_depth_aware=True,
    )
    depth_result = run_experiment(depth_config, compute_metrics=True, cleanup_decoded=False, sample_rate=10)

    # Decode frames
    base_decoded_dir = base_config.output_dir / f"{base_config.name}_decoded"
    depth_decoded_dir = depth_config.output_dir / f"{depth_config.name}_decoded"

    if not base_decoded_dir.exists():
        decode_to_frames(Path(base_result["encoding"]["output_path"]), base_decoded_dir)
    if not depth_decoded_dir.exists():
        decode_to_frames(Path(depth_result["encoding"]["output_path"]), depth_decoded_dir)

    # Create side-by-side frames
    comparison_frames_dir = COMPARISON_DIR / f"frames_crf{crf}"
    comparison_frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating side-by-side comparison frames...")
    for i in range(num_frames):
        ref_path = COLOR_FRAMES_DIR / f"frame_{i:03d}.png"
        base_path = base_decoded_dir / f"decoded_{i+1:03d}.png"
        depth_path = depth_decoded_dir / f"decoded_{i+1:03d}.png"
        output_path = comparison_frames_dir / f"compare_{i:04d}.png"

        if ref_path.exists() and base_path.exists() and depth_path.exists():
            create_side_by_side_frame(ref_path, base_path, depth_path, output_path, crf)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_frames} frames")

    # Encode comparison video
    output_video = COMPARISON_DIR / f"comparison_crf{crf}.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", str(comparison_frames_dir / "compare_%04d.png"),
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_video)
    ]

    print(f"Encoding comparison video...")
    subprocess.run(ffmpeg_cmd, capture_output=True)

    print(f"\n{'='*60}")
    print(f"Comparison video saved: {output_video}")
    print(f"{'='*60}")

    # Print metrics summary
    print(f"\nMetrics Summary:")
    print(f"  Baseline:    ROI PSNR = {base_result['metrics']['psnr_roi_mean']:.2f} dB, "
          f"Size = {base_result['encoding']['file_size_mb']:.2f} MB")
    print(f"  Depth-Aware: ROI PSNR = {depth_result['metrics']['psnr_roi_mean']:.2f} dB, "
          f"Size = {depth_result['encoding']['file_size_mb']:.2f} MB")
    print(f"  Improvement: +{depth_result['metrics']['psnr_roi_mean'] - base_result['metrics']['psnr_roi_mean']:.2f} dB")

    return {
        "baseline": base_result,
        "depth_aware": depth_result,
        "comparison_video": str(output_video),
    }


def create_crop_comparison_video(num_frames=500, crf=40, crop_region="center"):
    """Generate zoomed-in crop comparison to highlight detail differences."""
    print(f"\n{'='*60}")
    print(f"Generating crop comparison video ({num_frames} frames, CRF {crf})")
    print(f"{'='*60}")

    # Use existing decoded frames if available
    base_decoded_dir = OUTPUT_DIR / f"comparison_base_crf{crf}" / f"comparison_base_crf{crf}_decoded"
    depth_decoded_dir = OUTPUT_DIR / f"comparison_depth_crf{crf}" / f"comparison_depth_crf{crf}_decoded"

    if not base_decoded_dir.exists() or not depth_decoded_dir.exists():
        print("Please run create_comparison_video() first to generate decoded frames.")
        return None

    # Create cropped comparison frames
    crop_frames_dir = COMPARISON_DIR / f"crop_frames_crf{crf}"
    crop_frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating cropped comparison frames...")
    for i in range(num_frames):
        ref_path = COLOR_FRAMES_DIR / f"frame_{i:03d}.png"
        base_path = base_decoded_dir / f"decoded_{i+1:03d}.png"
        depth_path = depth_decoded_dir / f"decoded_{i+1:03d}.png"

        if not all(p.exists() for p in [ref_path, base_path, depth_path]):
            continue

        ref = Image.open(ref_path)
        base = Image.open(base_path)
        depth = Image.open(depth_path)

        w, h = ref.size
        # Center crop (where the fractal is)
        crop_size = min(w, h) // 2
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        crop_box = (left, top, left + crop_size, top + crop_size)

        ref_crop = ref.crop(crop_box).resize((512, 512), Image.Resampling.LANCZOS)
        base_crop = base.crop(crop_box).resize((512, 512), Image.Resampling.LANCZOS)
        depth_crop = depth.crop(crop_box).resize((512, 512), Image.Resampling.LANCZOS)

        # Create canvas
        canvas = Image.new('RGB', (512 * 3, 512 + 50), color=(30, 30, 30))
        canvas.paste(ref_crop, (0, 50))
        canvas.paste(base_crop, (512, 50))
        canvas.paste(depth_crop, (1024, 50))

        # Add labels
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        except:
            font = ImageFont.load_default()

        draw.text((200, 12), "Reference", fill=(255, 255, 255), font=font)
        draw.text((680, 12), f"Baseline", fill=(100, 180, 255), font=font)
        draw.text((1160, 12), f"Depth-Aware", fill=(100, 255, 150), font=font)

        canvas.save(crop_frames_dir / f"crop_{i:04d}.png")

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_frames} frames")

    # Encode crop comparison video
    output_video = COMPARISON_DIR / f"comparison_crop_crf{crf}.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", str(crop_frames_dir / "crop_%04d.png"),
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_video)
    ]

    subprocess.run(ffmpeg_cmd, capture_output=True)

    print(f"\nCrop comparison video saved: {output_video}")
    return str(output_video)


if __name__ == "__main__":
    # Generate full video comparison at CRF 40 (most visible difference)
    result = create_comparison_video(num_frames=500, crf=40)

    # Also create zoomed crop version
    create_crop_comparison_video(num_frames=500, crf=40)
