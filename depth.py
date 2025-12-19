"""
Depth-Aware Video Compression Pipeline - Depth Processing

This module converts depth maps into ROI regions for encoder-level QP allocation.

Key Concept:
------------
Instead of blurring/degrading far regions (destroying information), we generate
metadata that tells the encoder how to allocate bits:

    Depth Map → Importance Map → ROI Regions → FFmpeg addroi filter → Encoder

The encoder then makes smarter rate-distortion decisions, spending bits where
they matter most (near/foreground) while using fewer bits on less important
areas (far/background).
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image

from config import DepthROIConfig


@dataclass
class ROIRegion:
    """
    A rectangular region of interest with associated quality offset.

    FFmpeg's addroi filter expects:
    - Coordinates as fractions of frame dimensions [0.0, 1.0]
    - qoffset in range [-1.0, 1.0]:
        -1.0 = best quality (large negative QP offset)
         0.0 = no change
        +1.0 = worst quality (large positive QP offset)
    """
    x: float  # Left edge (0.0 - 1.0)
    y: float  # Top edge (0.0 - 1.0)
    w: float  # Width (0.0 - 1.0)
    h: float  # Height (0.0 - 1.0)
    qoffset: float  # Quality offset (-1.0 to 1.0)

    def to_addroi_params(self) -> str:
        """Generate FFmpeg addroi filter parameters for this region."""
        return f"{self.x:.4f}*iw:{self.y:.4f}*ih:{self.w:.4f}*iw:{self.h:.4f}*ih:{self.qoffset:.3f}"


def load_depth_map(path: Path) -> np.ndarray:
    """
    Load a depth map from file.

    Returns:
        2D numpy array with depth values normalized to [0, 1]
    """
    img = Image.open(path)
    depth = np.array(img, dtype=np.float32)

    # Handle RGB depth maps (take first channel or convert to grayscale)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    # Normalize to [0, 1]
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max > depth_min:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros_like(depth)

    return depth


def detect_depth_orientation(depth: np.ndarray) -> str:
    """
    Auto-detect whether near objects have high or low depth values.

    Heuristic: In most scenes, near objects (foreground) occupy less area
    than far objects (background). We check which interpretation gives
    a reasonable foreground percentage.
    """
    # Test both interpretations at 20th percentile
    threshold_low = np.percentile(depth, 20)
    threshold_high = np.percentile(depth, 80)

    near_low_fraction = np.mean(depth <= threshold_low)
    near_high_fraction = np.mean(depth >= threshold_high)

    # Foreground should typically be 10-40% of frame
    # Choose interpretation closer to 25% foreground
    target = 0.25
    if abs(near_low_fraction - target) < abs(near_high_fraction - target):
        return "near_low"
    else:
        return "near_high"


def create_importance_map(
    depth: np.ndarray,
    config: DepthROIConfig
) -> np.ndarray:
    """
    Convert depth map to importance map.

    Returns:
        2D array where 1.0 = most important (near), 0.0 = least important (far)
    """
    # Determine depth orientation
    if config.depth_mode == "auto":
        orientation = detect_depth_orientation(depth)
    else:
        orientation = config.depth_mode

    # Create importance based on orientation
    if orientation == "near_high":
        # Higher depth = nearer = more important
        importance = depth.copy()
    else:
        # Lower depth = nearer = more important
        importance = 1.0 - depth

    return importance


def importance_to_qoffset_map(
    importance: np.ndarray,
    config: DepthROIConfig
) -> np.ndarray:
    """
    Convert importance map to QP offset map with discrete quality levels.

    Maps importance [0, 1] to qoffset [far_qoffset, near_qoffset]
    High importance (1.0) → near_qoffset (negative = better quality)
    Low importance (0.0) → far_qoffset (positive = worse quality)

    With num_quality_levels=4:
        Level 0 (0-25% importance): far_qoffset (worst quality)
        Level 1 (25-50% importance): interpolated
        Level 2 (50-75% importance): interpolated
        Level 3 (75-100% importance): near_qoffset (best quality)
    """
    num_levels = config.num_quality_levels

    # Quantize importance into discrete levels
    # importance [0,1] -> level [0, num_levels-1]
    levels = np.floor(importance * num_levels).astype(np.int32)
    levels = np.clip(levels, 0, num_levels - 1)

    # Map each level to a qoffset value
    # Level 0 -> far_qoffset, Level (num_levels-1) -> near_qoffset
    qoffset_values = np.linspace(config.far_qoffset, config.near_qoffset, num_levels)

    # Create output map
    qoffset = qoffset_values[levels]

    return qoffset


def get_quality_level_info(config: DepthROIConfig) -> list[dict]:
    """
    Get information about each quality level for visualization/debugging.

    Returns list of dicts with level info: index, importance_range, qoffset, label
    """
    num_levels = config.num_quality_levels
    qoffset_values = np.linspace(config.far_qoffset, config.near_qoffset, num_levels)

    # Quality descriptions for different level counts
    desc_map = {
        2: ["Lowest", "Highest"],
        3: ["Lowest", "Medium", "Highest"],
        4: ["Lowest", "Low", "High", "Highest"],
        5: ["Lowest", "Low", "Medium", "High", "Highest"],
        6: ["Lowest", "Low", "Medium-Low", "Medium-High", "High", "Highest"],
    }
    descriptions = desc_map.get(num_levels, [f"Level {i}" for i in range(num_levels)])

    levels = []
    for i in range(num_levels):
        imp_min = i / num_levels
        imp_max = (i + 1) / num_levels
        levels.append({
            "level": i,
            "importance_range": (imp_min, imp_max),
            "qoffset": float(qoffset_values[i]),
            "label": f"Q{i}",
            "quality_desc": descriptions[i]
        })
    return levels


def extract_roi_regions(
    qoffset_map: np.ndarray,
    config: DepthROIConfig,  # Reserved for future filtering options
    grid_size: int = 16
) -> list[ROIRegion]:
    """
    Convert a QP offset map into discrete ROI regions.

    FFmpeg's addroi filter works with rectangular regions. We divide the
    frame into a grid and compute average qoffset per cell.

    Args:
        qoffset_map: 2D array of QP offsets
        config: ROI configuration
        grid_size: Number of cells per dimension (16x16 = 256 regions)

    Returns:
        List of ROI regions with their quality offsets
    """
    h, w = qoffset_map.shape
    cell_h = h // grid_size
    cell_w = w // grid_size

    regions = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            # Extract cell
            y0 = gy * cell_h
            y1 = (gy + 1) * cell_h if gy < grid_size - 1 else h
            x0 = gx * cell_w
            x1 = (gx + 1) * cell_w if gx < grid_size - 1 else w

            cell = qoffset_map[y0:y1, x0:x1]
            avg_qoffset = float(np.mean(cell))

            region = ROIRegion(
                x=x0 / w,
                y=y0 / h,
                w=(x1 - x0) / w,
                h=(y1 - y0) / h,
                qoffset=avg_qoffset
            )
            regions.append(region)

    return regions


def depth_to_roi_regions(
    depth_path: Path,
    config: DepthROIConfig,
    grid_size: int = 8
) -> list[ROIRegion]:
    """
    Complete pipeline: depth map file → list of ROI regions.

    This is the main entry point for depth-aware encoding.
    """
    depth = load_depth_map(depth_path)
    importance = create_importance_map(depth, config)
    qoffset_map = importance_to_qoffset_map(importance, config)
    regions = extract_roi_regions(qoffset_map, config, grid_size)
    return regions


def generate_addroi_filter(regions: list[ROIRegion]) -> str:
    """
    Generate FFmpeg filter string for multiple ROI regions.

    Example output:
        addroi=0.0:0.0:0.5:0.5:-0.3,addroi=0.5:0.5:0.5:0.5:0.3

    Note: FFmpeg applies ROIs additively, so overlapping regions
    combine their offsets.
    """
    if not regions:
        return ""

    filters = []
    for region in regions:
        params = region.to_addroi_params()
        filters.append(f"addroi={params}")

    return ",".join(filters)


# ============================================================================
# Analysis Utilities
# ============================================================================

def analyze_depth_map(depth_path: Path) -> dict:
    """Analyze a depth map and return statistics."""
    depth = load_depth_map(depth_path)

    return {
        "shape": depth.shape,
        "min": float(depth.min()),
        "max": float(depth.max()),
        "mean": float(depth.mean()),
        "std": float(depth.std()),
        "percentiles": {
            10: float(np.percentile(depth, 10)),
            25: float(np.percentile(depth, 25)),
            50: float(np.percentile(depth, 50)),
            75: float(np.percentile(depth, 75)),
            90: float(np.percentile(depth, 90)),
        },
        "detected_orientation": detect_depth_orientation(depth),
    }


if __name__ == "__main__":
    # Quick test with first depth frame
    from config import DEPTH_FRAMES_DIR, DepthROIConfig

    depth_path = DEPTH_FRAMES_DIR / "depth_000.png"
    if depth_path.exists():
        print("Analyzing depth map...")
        stats = analyze_depth_map(depth_path)
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\nGenerating ROI regions...")
        config = DepthROIConfig()
        regions = depth_to_roi_regions(depth_path, config)
        print(f"  Generated {len(regions)} ROI regions")

        if regions:
            print(f"\nSample region: {regions[0]}")
            print(f"FFmpeg filter (first 3 regions):")
            print(f"  {generate_addroi_filter(regions[:3])}")
