"""
RGBD Video Compression - Utilities

Core utilities for loading, processing, and encoding frames.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import subprocess
from typing import Iterator

from config import COLOR_FRAMES_DIR, DEPTH_FRAMES_DIR, FRAME_SIZE


def get_color_frame_path(frame_idx: int) -> Path:
    """Get path to color frame."""
    return COLOR_FRAMES_DIR / f"frame_{frame_idx:03d}.png"


def get_depth_frame_path(frame_idx: int) -> Path:
    """Get path to depth frame."""
    return DEPTH_FRAMES_DIR / f"depth_{frame_idx:03d}.png"


def load_color_frame(frame_idx: int) -> np.ndarray:
    """Load RGB color frame as numpy array."""
    path = get_color_frame_path(frame_idx)
    if not path.exists():
        raise FileNotFoundError(f"Color frame not found: {path}")
    return np.array(Image.open(path).convert("RGB"))


def load_depth_frame(frame_idx: int) -> np.ndarray:
    """Load depth map as normalized float array [0, 1]."""
    path = get_depth_frame_path(frame_idx)
    if not path.exists():
        raise FileNotFoundError(f"Depth frame not found: {path}")

    depth = np.array(Image.open(path))

    # Normalize to [0, 1]
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 65535.0
    elif depth.dtype == np.uint8:
        depth = depth.astype(np.float32) / 255.0
    else:
        depth = depth.astype(np.float32)
        if depth.max() > 1.0:
            depth = depth / depth.max()

    return depth


def depth_to_importance(depth: np.ndarray, mode: str = "near_high") -> np.ndarray:
    """
    Convert depth map to importance map.

    Args:
        depth: Normalized depth map [0, 1]
        mode: 'near_high' = high depth values are important (near objects)
              'near_low' = low depth values are important

    Returns:
        Importance map [0, 1] where 1 = most important
    """
    if mode == "near_high":
        # High depth = near = important
        importance = depth.copy()
    else:
        # Low depth = near = important
        importance = 1.0 - depth

    # Handle zero/background regions (usually black in Mandelbulb)
    # These should have low importance
    background_mask = depth < 0.01
    importance[background_mask] = 0.0

    return importance


def iter_frames(start: int, end: int) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """
    Iterate over frame pairs (color, depth).

    Yields:
        (frame_idx, color_array, depth_array)
    """
    for i in range(start, end):
        try:
            color = load_color_frame(i)
            depth = load_depth_frame(i)
            yield i, color, depth
        except FileNotFoundError:
            continue


def save_frame(frame: np.ndarray, path: Path) -> None:
    """Save numpy array as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.astype(np.uint8)).save(path)


def run_ffmpeg(cmd: list[str], desc: str = "") -> subprocess.CompletedProcess:
    """Run FFmpeg with error handling."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error ({desc}): {e.stderr}")
        raise


def encode_frames_to_video(
    frames_dir: Path,
    output_path: Path,
    fps: int = 30,
    start_number: int = 0,
    num_frames: int = None,
    encoder_args: list[str] = None,
    frame_pattern: str = "frame_%03d.png"
) -> int:
    """
    Encode PNG frames to video.

    Args:
        frames_dir: Directory containing frames
        output_path: Output video path
        fps: Frames per second
        start_number: Starting frame number
        num_frames: Number of frames to encode
        encoder_args: FFmpeg encoder arguments
        frame_pattern: Frame filename pattern (default: frame_%03d.png)

    Returns:
        File size in bytes
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", str(start_number),
        "-i", str(frames_dir / frame_pattern),
    ]

    if num_frames:
        cmd.extend(["-frames:v", str(num_frames)])

    if encoder_args:
        cmd.extend(encoder_args)
    else:
        cmd.extend(["-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p"])

    cmd.append(str(output_path))

    run_ffmpeg(cmd, "encoding")
    return output_path.stat().st_size


def get_video_size_mb(path: Path) -> float:
    """Get video file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def compute_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    """Compute SSIM between two images (simplified version)."""
    # Use skimage if available, otherwise simple approximation
    try:
        from skimage.metrics import structural_similarity
        if original.ndim == 3:
            return structural_similarity(original, compressed, channel_axis=2)
        return structural_similarity(original, compressed)
    except ImportError:
        # Simple SSIM approximation
        c1, c2 = 6.5025, 58.5225
        mu1, mu2 = original.mean(), compressed.mean()
        sigma1_sq = original.var()
        sigma2_sq = compressed.var()
        sigma12 = np.mean((original - mu1) * (compressed - mu2))

        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        return float(ssim)
