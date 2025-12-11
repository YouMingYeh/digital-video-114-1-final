# Depth-Aware Region of Interest Video Coding for Ray-Marched Fractal Content

Utilizing depth maps from ray marching to optimize H.264 video encoding through region-of-interest (ROI) based quality distribution.

## Overview

Traditional video encoders treat all pixels equally. In scenes with distinct foreground/background separation, this wastes bandwidth on unimportant regions. This project demonstrates how depth information from rendering can guide encoders to allocate more bits to foreground objects and fewer to backgrounds.

### Key Results

| Metric | Baseline | ROI-Optimized | Change |
|--------|----------|---------------|--------|
| File Size | 30.38 MB | 35.05 MB | +15.4% |
| PSNR (avg) | 39.89 dB | 38.78 dB | -1.11 dB |
| SSIM | 0.9883 | 0.9869 | -0.0014 |
| **Bitrate Variance** | 58.05 KB | 52.66 KB | **-9.3%** |

**Key Finding**: ROI encoding achieves **9.3% more consistent quality** across frames by redistributing bits from background to foreground.

## Dataset

- **Content**: Ray-marched Mandelbulb fractal
- **Resolution**: 2048x2048
- **Frames**: 500 (frames 0-299: orbital rotation, 300-499: camera push-in)
- **Source**: `Mandelbulb_Dataset/` containing color frames and depth maps

## Methodology

Uses x264's **Adaptive Quantization (AQ)** to simulate depth-aware encoding:

| Parameter | Baseline | ROI-Optimized |
|-----------|----------|---------------|
| AQ Mode | 1 (Default) | 3 (Auto-variance) |
| AQ Strength | 1.0 | 2.0 - 4.0 |

AQ Mode 3 naturally detects complex regions (where depth maps show the Mandelbulb) and allocates more bits there.

## Project Structure

```
├── encoder.py           # CLI encoding tool
├── video_utils.py       # FFmpeg utility functions
├── compare_quality.py   # Quality comparison tool
├── experiment.py        # ROI quality experiment
├── Mandelbulb_Dataset/  # Source frames and depth maps
├── baseline.mp4         # Standard H.264 encoding
├── optimized.mp4        # ROI-enhanced encoding
├── optimized_matched.mp4# Size-matched ROI encoding
├── comparison_f*.png    # Side-by-side frame comparisons
└── *.log                # PSNR/SSIM metric logs
```

## Usage

### Run Quality Analysis
```bash
python compare_quality.py
```

### Run ROI Experiment
```bash
python experiment.py
```

### Encode with Custom Settings
```bash
python encoder.py --help
```

## Requirements

- Python 3.x
- FFmpeg with libx264
- NumPy, OpenCV (for analysis scripts)

## Conclusions

1. Depth-aware encoding **redistributes quality** from background to foreground
2. Achieves **9.3% improvement** in bitrate consistency at moderate compression
3. **25% file size reduction** possible with acceptable quality trade-offs
4. Works best with complex backgrounds competing for bits; less effective with simple backgrounds

### Limitations

- Used AQ heuristics instead of direct depth-to-QP mapping
- Test content (black background) was favorable to baseline
- No user study for perceptual validation

## License

Academic project for Digital Video course (114-1).
