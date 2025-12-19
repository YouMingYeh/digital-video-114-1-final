"""
Depth-Aware Video Compression Pipeline - Encoder

This module wraps FFmpeg for video encoding with depth-aware ROI support.

Key Features:
-------------
1. Standard encoding (baseline for comparison)
2. Depth-aware encoding using FFmpeg's addroi filter
3. Per-frame ROI generation from depth maps
4. Support for x264, x265, and AV1 codecs
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from config import (
    ExperimentConfig,
    EncoderConfig,
    COLOR_FRAMES_DIR,
    DEPTH_FRAMES_DIR,
)
from depth import depth_to_roi_regions, generate_addroi_filter, DepthROIConfig


def get_frame_paths(start: int, count: int) -> tuple[list[Path], list[Path]]:
    """Get lists of color and depth frame paths."""
    color_paths = []
    depth_paths = []

    for i in range(start, start + count):
        color_path = COLOR_FRAMES_DIR / f"frame_{i:03d}.png"
        depth_path = DEPTH_FRAMES_DIR / f"depth_{i:03d}.png"
        color_paths.append(color_path)
        depth_paths.append(depth_path)

    return color_paths, depth_paths


def run_ffmpeg(cmd: list[str], description: str = "") -> subprocess.CompletedProcess:
    """Run FFmpeg command with error handling."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error during {description}:")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Stderr: {e.stderr}")
        raise


def encode_baseline(
    config: ExperimentConfig,
    output_path: Path
) -> dict:
    """
    Encode video without depth awareness (baseline).

    This uses standard FFmpeg encoding with the configured encoder settings
    but no ROI-based bit allocation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build FFmpeg command
    # Input: image sequence
    input_pattern = str(COLOR_FRAMES_DIR / "frame_%03d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(config.fps),
        "-start_number", str(config.start_frame),
        "-i", input_pattern,
        "-frames:v", str(config.num_frames),
    ]

    # Add encoder parameters
    cmd.extend(config.encoder.to_ffmpeg_params())

    # Output
    cmd.append(str(output_path))

    run_ffmpeg(cmd, "baseline encoding")

    # Get file size
    file_size = output_path.stat().st_size

    return {
        "output_path": str(output_path),
        "file_size_bytes": file_size,
        "file_size_mb": file_size / (1024 * 1024),
        "method": "baseline",
    }


def encode_with_depth_roi(
    config: ExperimentConfig,
    output_path: Path,
    grid_size: int = 8
) -> dict:
    """
    Encode video with depth-aware ROI-based bit allocation.

    This is the core innovation:
    1. For each frame, load depth map
    2. Convert depth → importance → QP offsets → ROI regions
    3. Generate FFmpeg filter graph with addroi filters
    4. Encode with ROI metadata guiding encoder decisions

    The encoder receives hints about which regions need quality preservation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate ROI filter chain for each frame
    # FFmpeg's addroi filter works per-frame when combined with sendcmd
    # For simplicity, we'll use a representative frame's ROI for the whole video
    # (In production, you'd use sendcmd/zmq for per-frame ROI updates)

    # Get middle frame's depth for representative ROI
    mid_frame = config.start_frame + config.num_frames // 2
    depth_path = DEPTH_FRAMES_DIR / f"depth_{mid_frame:03d}.png"

    if not depth_path.exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    # Generate ROI regions from depth
    regions = depth_to_roi_regions(depth_path, config.depth_roi, grid_size)

    # Build filter string
    roi_filter = generate_addroi_filter(regions)

    # Build FFmpeg command
    input_pattern = str(COLOR_FRAMES_DIR / "frame_%03d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(config.fps),
        "-start_number", str(config.start_frame),
        "-i", input_pattern,
        "-frames:v", str(config.num_frames),
    ]

    # Add ROI filter if we have regions
    if roi_filter:
        cmd.extend(["-vf", roi_filter])

    # Add encoder parameters
    cmd.extend(config.encoder.to_ffmpeg_params())

    # Output
    cmd.append(str(output_path))

    run_ffmpeg(cmd, "depth-aware ROI encoding")

    # Get file size
    file_size = output_path.stat().st_size

    return {
        "output_path": str(output_path),
        "file_size_bytes": file_size,
        "file_size_mb": file_size / (1024 * 1024),
        "method": "depth_aware",
        "num_roi_regions": len(regions),
        "roi_filter": roi_filter[:200] + "..." if len(roi_filter) > 200 else roi_filter,
    }


def encode_with_perframe_roi(
    config: ExperimentConfig,
    output_path: Path,
    grid_size: int = 8
) -> dict:
    """
    Advanced: Encode with per-frame ROI updates.

    This is more accurate for dynamic scenes but slower.
    Uses FFmpeg's filtergraph with per-frame ROI calculation.

    Note: This requires encoding in two passes or using complex filtergraphs.
    For research purposes, the single-ROI approach is often sufficient
    if the scene depth distribution doesn't change dramatically.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strategy: Generate ROI data for each frame, then encode
    # We'll create a temporary script for sendcmd

    _, depth_paths = get_frame_paths(config.start_frame, config.num_frames)

    # Collect all unique ROI configurations
    # (This is a simplified version - full implementation would use sendcmd)

    # For now, compute average ROI across all frames
    from depth import load_depth_map, create_importance_map, importance_to_qoffset_map

    all_qoffsets = []
    for dp in depth_paths:
        if dp.exists():
            depth = load_depth_map(dp)
            importance = create_importance_map(depth, config.depth_roi)
            qoffset = importance_to_qoffset_map(importance, config.depth_roi)
            all_qoffsets.append(qoffset)

    if all_qoffsets:
        # Average QP offset map across frames
        avg_qoffset = sum(all_qoffsets) / len(all_qoffsets)

        from depth import extract_roi_regions
        regions = extract_roi_regions(avg_qoffset, config.depth_roi, grid_size)
        roi_filter = generate_addroi_filter(regions)
    else:
        roi_filter = ""
        regions = []

    # Build FFmpeg command
    input_pattern = str(COLOR_FRAMES_DIR / "frame_%03d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(config.fps),
        "-start_number", str(config.start_frame),
        "-i", input_pattern,
        "-frames:v", str(config.num_frames),
    ]

    if roi_filter:
        cmd.extend(["-vf", roi_filter])

    cmd.extend(config.encoder.to_ffmpeg_params())
    cmd.append(str(output_path))

    run_ffmpeg(cmd, "per-frame ROI encoding")

    file_size = output_path.stat().st_size

    return {
        "output_path": str(output_path),
        "file_size_bytes": file_size,
        "file_size_mb": file_size / (1024 * 1024),
        "method": "perframe_roi",
        "num_roi_regions": len(regions),
        "num_frames_analyzed": len(all_qoffsets),
    }


def encode(config: ExperimentConfig, output_path: Optional[Path] = None) -> dict:
    """
    Main encoding entry point.

    Automatically selects encoding method based on configuration.
    """
    if output_path is None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = config.output_dir / f"{config.name}.mp4"

    if config.use_depth_aware:
        return encode_with_depth_roi(config, output_path)
    else:
        return encode_baseline(config, output_path)


def decode_to_frames(
    video_path: Path,
    output_dir: Path,
    frame_pattern: str = "decoded_%03d.png"
) -> list[Path]:
    """
    Decode video back to frames for quality comparison.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        str(output_dir / frame_pattern)
    ]

    run_ffmpeg(cmd, "decoding")

    # Return list of decoded frame paths
    frames = sorted(output_dir.glob("decoded_*.png"))
    return frames


def get_video_info(video_path: Path) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    # Extract key metrics
    video_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
        {}
    )

    format_info = info.get("format", {})

    return {
        "codec": video_stream.get("codec_name"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "duration_sec": float(format_info.get("duration", 0)),
        "bitrate_kbps": int(format_info.get("bit_rate", 0)) / 1000,
        "file_size_bytes": int(format_info.get("size", 0)),
    }


if __name__ == "__main__":
    from config import baseline_config, depth_aware_config

    print("Testing encoder module...\n")

    # Test baseline encoding
    print("=" * 60)
    print("Baseline Encoding (no depth awareness)")
    print("=" * 60)
    baseline = baseline_config(num_frames=30, crf=23)
    result = encode(baseline)
    print(f"  Output: {result['output_path']}")
    print(f"  Size: {result['file_size_mb']:.2f} MB")
    print(f"  Method: {result['method']}")

    # Test depth-aware encoding
    print("\n" + "=" * 60)
    print("Depth-Aware ROI Encoding")
    print("=" * 60)
    depth = depth_aware_config(num_frames=30, crf=23)
    result = encode(depth)
    print(f"  Output: {result['output_path']}")
    print(f"  Size: {result['file_size_mb']:.2f} MB")
    print(f"  Method: {result['method']}")
    print(f"  ROI Regions: {result.get('num_roi_regions', 'N/A')}")
