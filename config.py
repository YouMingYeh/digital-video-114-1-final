"""
RGBD Video Compression - Configuration

Fresh implementation based on research:
- N-DEPTH neural encoding approach
- 3D-HEVC joint coding concepts
- Depth map packing for standard codecs
"""

from pathlib import Path
from dataclasses import dataclass

# Dataset paths
PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / "Mandelbulb_Dataset"
COLOR_FRAMES_DIR = DATASET_DIR / "output_color"
DEPTH_FRAMES_DIR = DATASET_DIR / "output_depth"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Dataset info (from README)
TOTAL_FRAMES = 500
ROTATION_FRAMES = (0, 300)   # Frames 0-299: rotation, stable depth
ZOOM_FRAMES = (300, 500)     # Frames 300-499: zoom, changing depth
FRAME_SIZE = (2048, 2048)
FPS = 30


@dataclass
class EncoderConfig:
    """Video encoder settings."""
    codec: str = "libx264"
    crf: int = 23
    preset: str = "medium"

    def to_ffmpeg_args(self) -> list[str]:
        return [
            "-c:v", self.codec,
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-pix_fmt", "yuv420p",
        ]


@dataclass
class DepthConfig:
    """Depth processing settings."""
    # Depth interpretation: 'near_high' means high values = near (more important)
    depth_mode: str = "near_high"

    # Quality allocation
    foreground_quality: int = 90   # JPEG quality for near/important regions
    background_quality: int = 30   # JPEG quality for far/unimportant regions

    # Thresholds for importance classification
    importance_levels: int = 5     # Number of quality levels
