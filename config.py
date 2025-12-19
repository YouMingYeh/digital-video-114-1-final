"""
Depth-Aware Video Compression Pipeline - Configuration

This module defines all configuration for the depth-aware encoding experiments.
The key insight: we use depth maps to generate QP offsets that guide the encoder's
rate-distortion decisions, rather than pre-degrading the source content.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ============================================================================
# Path Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "Mandelbulb_Dataset"
COLOR_FRAMES_DIR = DATASET_DIR / "output_color"
DEPTH_FRAMES_DIR = DATASET_DIR / "output_depth"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ============================================================================
# Encoder Configuration
# ============================================================================

@dataclass
class EncoderConfig:
    """Configuration for video encoder settings."""

    codec: Literal["libx264", "libx265", "libsvtav1"] = "libx264"
    preset: str = "ultrafast"  # Encoding speed preset (ultrafast for testing)

    # Rate control mode: "crf" (quality) or "abr" (bitrate)
    rate_mode: Literal["crf", "abr"] = "crf"
    crf: int = 23  # For CRF mode: 0-51, lower = better quality
    bitrate_kbps: int = 5000  # For ABR mode: target bitrate in kbps

    # x264/x265 specific: Adaptive Quantization
    aq_mode: int = 1  # 0=off, 1=variance, 2=complexity, 3=motion
    aq_strength: float = 1.0  # 0.5-2.0, higher = more AQ

    # Pixel format
    pix_fmt: str = "yuv420p"

    def to_ffmpeg_params(self) -> list[str]:
        """Convert config to FFmpeg command-line parameters."""
        params = [
            "-c:v", self.codec,
            "-preset", self.preset,
            "-pix_fmt", self.pix_fmt,
        ]

        # Rate control
        if self.rate_mode == "crf":
            params.extend(["-crf", str(self.crf)])
        else:  # ABR mode
            params.extend(["-b:v", f"{self.bitrate_kbps}k"])

        if self.codec == "libx264":
            x264_opts = f"aq-mode={self.aq_mode}:aq-strength={self.aq_strength}"
            params.extend(["-x264-params", x264_opts])
        elif self.codec == "libx265":
            x265_opts = f"aq-mode={self.aq_mode}:aq-strength={self.aq_strength}"
            params.extend(["-x265-params", x265_opts])

        return params


# ============================================================================
# Depth-Aware ROI Configuration
# ============================================================================

@dataclass
class DepthROIConfig:
    """
    Configuration for depth-based Region of Interest encoding.

    The core idea: convert depth values to QP offsets.
    - Near regions (important): qp_offset < 0 → more bits, better quality
    - Far regions (less important): qp_offset > 0 → fewer bits, lower quality
    """

    # How to interpret depth values
    # "near_high": higher depth values = closer to camera (our Mandelbulb dataset)
    # "near_low": lower depth values = closer to camera (typical depth sensors)
    # Default to "near_high" for Mandelbulb where the fractal object has high depth
    depth_mode: Literal["near_high", "near_low", "auto"] = "near_high"

    # Number of discrete quality levels (2-8)
    # 2 = binary (high/low), 5 = recommended for smooth gradation
    num_quality_levels: int = 5

    # QP offset range for ROI encoding
    # FFmpeg addroi uses normalized [-1, 1] where:
    #   -1.0 = best quality (lowest QP offset)
    #    0.0 = no change
    #   +1.0 = worst quality (highest QP offset)
    #
    # Strategy: Keep ROI at baseline quality, reduce background quality
    # This redistributes bits rather than just adding more
    near_qoffset: float = -0.4   # Best quality for nearest regions
    far_qoffset: float = 0.4    # Reduced quality for far regions (save bits)

    # Depth thresholding
    near_percentile: float = 30.0  # Top 30% closest pixels = "near"
    far_percentile: float = 70.0   # Bottom 30% furthest pixels = "far"

    # Mask processing
    feather_radius: int = 8  # Gaussian blur radius for soft transitions

    # Minimum region size (as fraction of frame area)
    min_region_fraction: float = 0.01


@dataclass
class ExperimentConfig:
    """Configuration for a complete encoding experiment."""

    name: str = "experiment"

    # Frame range to process
    start_frame: int = 0
    num_frames: int = 120  # Use subset for quick tests
    fps: int = 30

    # Encoder settings
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    # Depth-aware settings
    depth_roi: DepthROIConfig = field(default_factory=DepthROIConfig)

    # Whether to use depth-aware encoding
    use_depth_aware: bool = True

    @property
    def output_dir(self) -> Path:
        return OUTPUT_DIR / self.name


# ============================================================================
# Preset Configurations
# ============================================================================

def baseline_config(num_frames: int = 120, crf: int = 23) -> ExperimentConfig:
    """Standard encoding without depth awareness."""
    return ExperimentConfig(
        name=f"baseline_crf{crf}",
        num_frames=num_frames,
        encoder=EncoderConfig(crf=crf, aq_mode=1),
        use_depth_aware=False,
    )


def depth_aware_config(num_frames: int = 120, crf: int = 23) -> ExperimentConfig:
    """Depth-aware encoding with 5-level QP allocation."""
    return ExperimentConfig(
        name=f"depth_aware_crf{crf}",
        num_frames=num_frames,
        encoder=EncoderConfig(crf=crf, aq_mode=1),
        depth_roi=DepthROIConfig(
            depth_mode="near_high",  # Mandelbulb: high depth = foreground object
            num_quality_levels=5,    # 5 discrete quality levels
            near_qoffset=-0.4,  # Best quality for ROI (more bits)
            far_qoffset=0.4,    # Reduced quality for background (fewer bits)
        ),
        use_depth_aware=True,
    )


def aggressive_depth_config(num_frames: int = 120, crf: int = 23) -> ExperimentConfig:
    """Aggressive depth-aware encoding - maximize ROI quality improvement."""
    return ExperimentConfig(
        name=f"depth_aggressive_crf{crf}",
        num_frames=num_frames,
        encoder=EncoderConfig(crf=crf, aq_mode=1),
        depth_roi=DepthROIConfig(
            depth_mode="near_high",  # Mandelbulb: high depth = foreground object
            num_quality_levels=5,    # 5 discrete quality levels
            near_qoffset=-0.6,  # Much better quality for ROI
            far_qoffset=0.6,    # Much worse quality for background
            near_percentile=15.0,  # Tighter ROI definition
            far_percentile=85.0,
        ),
        use_depth_aware=True,
    )
