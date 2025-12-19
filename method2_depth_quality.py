"""
Method 2: Depth-Guided Per-Frame Quality Preprocessing

Concept: Apply spatially-varying quality compression to each frame based on
depth/importance BEFORE video encoding. This truly reduces information in
unimportant regions at the frame level.

Approach:
1. Create importance map from depth
2. Apply region-based JPEG compression with varying quality
3. Encode pre-processed frames as video

This works because:
- JPEG compression is applied per-region with different quality levels
- Important regions (near) get high quality, unimportant (far) get low quality
- The video encoder then sees already-simplified background, naturally allocating fewer bits
"""

import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from io import BytesIO

from config import OUTPUT_DIR, EncoderConfig, DepthConfig
from utils import (
    iter_frames, save_frame, encode_frames_to_video,
    get_video_size_mb, depth_to_importance
)


def apply_depth_quality_jpeg(
    color: np.ndarray,
    depth: np.ndarray,
    depth_config: DepthConfig,
    block_size: int = 64
) -> np.ndarray:
    """
    Apply block-wise JPEG compression with depth-based quality.

    Args:
        color: RGB frame
        depth: Depth map [0, 1]
        depth_config: Quality settings
        block_size: Size of blocks for quality assignment

    Returns:
        Quality-processed frame
    """
    h, w = color.shape[:2]
    importance = depth_to_importance(depth, depth_config.depth_mode)

    # Quantize importance to quality levels
    num_levels = depth_config.importance_levels
    quality_range = depth_config.foreground_quality - depth_config.background_quality

    # Create block-wise quality map
    result = color.copy()

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)

            # Get average importance for this block
            block_importance = importance[y:y_end, x:x_end].mean()

            # Map importance to JPEG quality
            quality = int(
                depth_config.background_quality +
                quality_range * block_importance
            )

            # Extract and compress block
            block = color[y:y_end, x:x_end]

            # JPEG compress with varying quality
            pil_block = Image.fromarray(block)
            buffer = BytesIO()
            pil_block.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            compressed_block = np.array(Image.open(buffer))

            result[y:y_end, x:x_end] = compressed_block

    return result


def apply_depth_quality_blur(
    color: np.ndarray,
    depth: np.ndarray,
    depth_config: DepthConfig,
    max_blur: int = 15
) -> np.ndarray:
    """
    Alternative: Apply depth-weighted Gaussian blur.

    Near regions stay sharp, far regions get blurred.
    Simpler than JPEG blocks and often more effective.
    """
    importance = depth_to_importance(depth, depth_config.depth_mode)

    # Create blur pyramid
    blurred_versions = []
    for k in [3, 7, 11, 15]:
        blurred = cv2.GaussianBlur(color, (k, k), 0)
        blurred_versions.append(blurred)

    # Map importance to blur level
    num_levels = len(blurred_versions) + 1  # +1 for original
    level_map = ((1 - importance) * (num_levels - 0.001)).astype(np.int32)
    level_map = np.clip(level_map, 0, num_levels - 1)

    # Build result by selecting from blur levels
    result = color.copy()

    for level_idx, blurred in enumerate(blurred_versions):
        mask = (level_map == level_idx + 1)
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]
        result = np.where(mask, blurred, result)

    return result


def apply_depth_quality_bilateral(
    color: np.ndarray,
    depth: np.ndarray,
    depth_config: DepthConfig,
) -> np.ndarray:
    """
    Apply depth-weighted bilateral filter.

    Bilateral filter preserves edges while smoothing - ideal for maintaining
    fractal structure while reducing texture detail in background.
    """
    importance = depth_to_importance(depth, depth_config.depth_mode)

    # Strong bilateral filter
    filtered = cv2.bilateralFilter(color, 15, 150, 150)

    # Blend based on importance (smooth transition)
    # High importance = original, low importance = filtered
    blend = importance[:, :, np.newaxis]  # Expand to 3 channels

    result = (blend * color + (1 - blend) * filtered).astype(np.uint8)
    return result


def encode_depth_quality(
    start_frame: int,
    num_frames: int,
    output_dir: Path,
    encoder_config: EncoderConfig,
    depth_config: DepthConfig,
    method: str = "bilateral"
) -> dict:
    """
    Encode video with depth-guided quality preprocessing.

    Args:
        method: "jpeg", "blur", or "bilateral"
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / f"preprocessed_{method}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    process_func = {
        "jpeg": apply_depth_quality_jpeg,
        "blur": apply_depth_quality_blur,
        "bilateral": apply_depth_quality_bilateral,
    }[method]

    print(f"Preprocessing frames with {method} method...")

    for idx, color, depth in iter_frames(start_frame, start_frame + num_frames):
        processed = process_func(color, depth, depth_config)
        save_frame(processed, frames_dir / f"frame_{idx:03d}.png")

        if (idx - start_frame + 1) % 50 == 0:
            print(f"  Processed {idx - start_frame + 1}/{num_frames} frames")

    print(f"Encoding preprocessed video...")
    output_path = output_dir / f"depth_quality_{method}.mp4"
    encode_frames_to_video(
        frames_dir, output_path,
        start_number=start_frame,
        num_frames=num_frames,
        encoder_args=encoder_config.to_ffmpeg_args()
    )

    size_mb = get_video_size_mb(output_path)

    return {
        "method": f"depth_quality_{method}",
        "output_path": str(output_path),
        "size_mb": size_mb,
        "frames": num_frames,
        "preprocessing": method,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Method 2: Depth-Guided Quality Preprocessing")
    print("=" * 60)

    output_dir = OUTPUT_DIR / "method2_quality"
    encoder_config = EncoderConfig(crf=28, preset="medium")
    depth_config = DepthConfig(
        foreground_quality=95,
        background_quality=20,
    )

    for method in ["bilateral", "blur", "jpeg"]:
        print(f"\n--- Testing {method} ---")
        result = encode_depth_quality(
            start_frame=0,
            num_frames=100,
            output_dir=output_dir,
            encoder_config=encoder_config,
            depth_config=depth_config,
            method=method
        )
        print(f"Result: {result['size_mb']:.2f} MB")
