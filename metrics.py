"""
Depth-Aware Video Compression Pipeline - Quality Metrics

This module computes quality metrics for encoded videos, with special
emphasis on depth-stratified analysis (separate metrics for near vs far regions).

Key Metrics:
------------
- PSNR (Peak Signal-to-Noise Ratio): Higher = better, measures pixel-level fidelity
- SSIM (Structural Similarity): Higher = better, measures perceptual quality
- Region-specific metrics: PSNR/SSIM computed only within depth-defined ROI

The depth-stratified approach reveals whether the encoder allocated bits
effectively to perceptually important regions.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

from config import DEPTH_FRAMES_DIR, DepthROIConfig
from depth import load_depth_map, create_importance_map


@dataclass
class FrameMetrics:
    """Quality metrics for a single frame."""
    frame_index: int
    psnr_global: float
    psnr_roi: float  # Near/foreground region
    psnr_background: float  # Far/background region
    ssim_global: float
    roi_pixel_fraction: float  # What fraction of pixels are in ROI


@dataclass
class VideoMetrics:
    """Aggregate quality metrics for entire video."""
    num_frames: int

    # Global metrics (full frame)
    psnr_global_mean: float
    psnr_global_std: float
    ssim_global_mean: float

    # ROI metrics (near/foreground only)
    psnr_roi_mean: float
    psnr_roi_std: float

    # Background metrics (far/background only)
    psnr_background_mean: float
    psnr_background_std: float

    # ROI coverage
    roi_pixel_fraction_mean: float

    # Derived insights
    @property
    def roi_advantage_db(self) -> float:
        """How much better is ROI quality vs background (in dB)."""
        return self.psnr_roi_mean - self.psnr_background_mean


def compute_psnr(img1: np.ndarray, img2: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute PSNR between two images, optionally within a masked region.

    Args:
        img1, img2: Images as numpy arrays (same shape)
        mask: Optional binary mask (1 = include, 0 = exclude)

    Returns:
        PSNR in dB (higher is better)
    """
    # Ensure float for precision
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if mask is not None:
        # Expand mask to match image channels if needed
        if img1.ndim == 3 and mask.ndim == 2:
            mask = mask[:, :, np.newaxis]

        # Apply mask
        diff = (img1 - img2) * mask
        num_pixels = mask.sum()

        if num_pixels == 0:
            return float('inf')  # No pixels to compare

        mse = (diff ** 2).sum() / (num_pixels * (img1.shape[-1] if img1.ndim == 3 else 1))
    else:
        mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float('inf')  # Identical images

    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


def compute_ssim(img1: np.ndarray, img2: np.ndarray, window_size: int = 11) -> float:
    """
    Compute SSIM (Structural Similarity Index) between two images.

    Simplified implementation without scipy dependency.
    Uses the standard SSIM formula with default parameters.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Convert to grayscale if color
    if img1.ndim == 3:
        img1 = 0.299 * img1[:,:,0] + 0.587 * img1[:,:,1] + 0.114 * img1[:,:,2]
        img2 = 0.299 * img2[:,:,0] + 0.587 * img2[:,:,1] + 0.114 * img2[:,:,2]

    # SSIM constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Simple block-based SSIM
    block_size = 8
    h, w = img1.shape
    ssim_map = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block1 = img1[y:y+block_size, x:x+block_size]
            block2 = img2[y:y+block_size, x:x+block_size]

            mu1 = block1.mean()
            mu2 = block2.mean()
            sigma1_sq = block1.var()
            sigma2_sq = block2.var()
            sigma12 = ((block1 - mu1) * (block2 - mu2)).mean()

            ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim_map.append(ssim_val)

    return np.mean(ssim_map) if ssim_map else 0.0


def create_depth_mask(
    depth_path: Path,
    config: DepthROIConfig,
    is_roi: bool = True
) -> np.ndarray:
    """
    Create a binary mask from depth map.

    Args:
        depth_path: Path to depth map image
        config: ROI configuration
        is_roi: If True, mask for near/ROI regions. If False, mask for far/background.

    Returns:
        Binary mask (1 = included, 0 = excluded)
    """
    depth = load_depth_map(depth_path)
    importance = create_importance_map(depth, config)

    # For datasets with many zero-importance pixels (like Mandelbulb),
    # compute percentiles on non-zero values only
    nonzero_importance = importance[importance > 0.01]

    if len(nonzero_importance) == 0:
        # Fallback if no significant importance values
        nonzero_importance = importance.flatten()

    if is_roi:
        # High importance = near = ROI (top N% of non-zero importance)
        threshold = np.percentile(nonzero_importance, 100 - config.near_percentile)
        # Use > for strict inequality to handle edge cases
        mask = (importance > threshold).astype(np.float32)
    else:
        # Low importance = far = background
        threshold = np.percentile(importance, config.far_percentile)
        mask = (importance <= threshold).astype(np.float32)

    return mask


def compute_frame_metrics(
    ref_path: Path,
    dec_path: Path,
    depth_path: Path,
    frame_index: int,
    config: DepthROIConfig
) -> FrameMetrics:
    """
    Compute all quality metrics for a single frame.
    """
    # Load images
    ref_img = np.array(Image.open(ref_path).convert('RGB'))
    dec_img = np.array(Image.open(dec_path).convert('RGB'))

    # Ensure same size (decoded might be slightly different due to codec constraints)
    if ref_img.shape != dec_img.shape:
        # Resize decoded to match reference
        dec_pil = Image.open(dec_path).convert('RGB').resize(
            (ref_img.shape[1], ref_img.shape[0]),
            Image.Resampling.LANCZOS
        )
        dec_img = np.array(dec_pil)

    # Create masks
    roi_mask = create_depth_mask(depth_path, config, is_roi=True)
    bg_mask = create_depth_mask(depth_path, config, is_roi=False)

    # Compute metrics
    psnr_global = compute_psnr(ref_img, dec_img)
    psnr_roi = compute_psnr(ref_img, dec_img, roi_mask)
    psnr_background = compute_psnr(ref_img, dec_img, bg_mask)
    ssim_global = compute_ssim(ref_img, dec_img)

    roi_fraction = roi_mask.mean()

    return FrameMetrics(
        frame_index=frame_index,
        psnr_global=psnr_global,
        psnr_roi=psnr_roi,
        psnr_background=psnr_background,
        ssim_global=ssim_global,
        roi_pixel_fraction=roi_fraction,
    )


def compute_video_metrics(
    ref_frames: list[Path],
    dec_frames: list[Path],
    depth_frames: list[Path],
    config: DepthROIConfig,
    sample_rate: int = 1  # Process every Nth frame (1 = all, 5 = every 5th)
) -> VideoMetrics:
    """
    Compute aggregate metrics across frames.

    Args:
        sample_rate: Process every Nth frame for faster computation.
                    Use 1 for all frames, 5-10 for quick estimates.
    """
    frame_metrics = []

    # Sample frames for faster processing
    indices = range(0, len(ref_frames), sample_rate)

    for i in indices:
        if i >= len(ref_frames):
            break
        ref, dec, depth = ref_frames[i], dec_frames[i], depth_frames[i]
        if not all(p.exists() for p in [ref, dec, depth]):
            continue

        fm = compute_frame_metrics(ref, dec, depth, i, config)
        frame_metrics.append(fm)

    if not frame_metrics:
        raise ValueError("No valid frames to compute metrics")

    # Aggregate
    psnr_globals = [fm.psnr_global for fm in frame_metrics if fm.psnr_global != float('inf')]
    psnr_rois = [fm.psnr_roi for fm in frame_metrics if fm.psnr_roi != float('inf')]
    psnr_bgs = [fm.psnr_background for fm in frame_metrics if fm.psnr_background != float('inf')]
    ssim_globals = [fm.ssim_global for fm in frame_metrics]
    roi_fractions = [fm.roi_pixel_fraction for fm in frame_metrics]

    return VideoMetrics(
        num_frames=len(frame_metrics),
        psnr_global_mean=float(np.mean(psnr_globals)) if psnr_globals else 0.0,
        psnr_global_std=float(np.std(psnr_globals)) if psnr_globals else 0.0,
        ssim_global_mean=float(np.mean(ssim_globals)),
        psnr_roi_mean=float(np.mean(psnr_rois)) if psnr_rois else 0.0,
        psnr_roi_std=float(np.std(psnr_rois)) if psnr_rois else 0.0,
        psnr_background_mean=float(np.mean(psnr_bgs)) if psnr_bgs else 0.0,
        psnr_background_std=float(np.std(psnr_bgs)) if psnr_bgs else 0.0,
        roi_pixel_fraction_mean=float(np.mean(roi_fractions)),
    )


def compute_metrics_ffmpeg(
    ref_video: Path,
    enc_video: Path
) -> dict:
    """
    Use FFmpeg's built-in PSNR/SSIM filters for fast global metrics.

    This is faster than per-frame Python computation but doesn't support
    depth-stratified analysis.
    """
    # PSNR
    psnr_cmd = [
        "ffmpeg", "-i", str(enc_video), "-i", str(ref_video),
        "-lavfi", "psnr=stats_file=-",
        "-f", "null", "-"
    ]

    result = subprocess.run(psnr_cmd, capture_output=True, text=True)

    # Parse PSNR from stderr (FFmpeg outputs metrics there)
    psnr_avg = None
    for line in result.stderr.split('\n'):
        if 'average:' in line:
            # Format: "PSNR y:XX.XX u:XX.XX v:XX.XX average:XX.XX"
            parts = line.split('average:')
            if len(parts) > 1:
                try:
                    psnr_avg = float(parts[1].split()[0])
                except (ValueError, IndexError):
                    pass

    # SSIM
    ssim_cmd = [
        "ffmpeg", "-i", str(enc_video), "-i", str(ref_video),
        "-lavfi", "ssim=stats_file=-",
        "-f", "null", "-"
    ]

    result = subprocess.run(ssim_cmd, capture_output=True, text=True)

    ssim_avg = None
    for line in result.stderr.split('\n'):
        if 'All:' in line:
            parts = line.split('All:')
            if len(parts) > 1:
                try:
                    ssim_avg = float(parts[1].split()[0])
                except (ValueError, IndexError):
                    pass

    return {
        "psnr_global": psnr_avg,
        "ssim_global": ssim_avg,
    }


if __name__ == "__main__":
    from config import COLOR_FRAMES_DIR, DEPTH_FRAMES_DIR, DepthROIConfig

    print("Testing metrics module...\n")

    # Test with first frame (comparing to itself = perfect quality)
    ref_path = COLOR_FRAMES_DIR / "frame_000.png"
    depth_path = DEPTH_FRAMES_DIR / "depth_000.png"

    if ref_path.exists() and depth_path.exists():
        config = DepthROIConfig()

        print("Computing metrics (reference vs itself = perfect)...")
        fm = compute_frame_metrics(ref_path, ref_path, depth_path, 0, config)

        print(f"  PSNR Global: {fm.psnr_global:.2f} dB (should be inf)")
        print(f"  PSNR ROI: {fm.psnr_roi:.2f} dB")
        print(f"  PSNR Background: {fm.psnr_background:.2f} dB")
        print(f"  SSIM Global: {fm.ssim_global:.4f}")
        print(f"  ROI Fraction: {fm.roi_pixel_fraction:.2%}")
    else:
        print("Dataset not found - skipping metrics test")
