# Depth-Guided Video Compression: An Empirical Analysis

This project investigates whether depth information can improve video compression efficiency through preprocessing. **Spoiler: It doesn't.**

## Key Finding

Using rigorous BD-Rate evaluation, we found that depth-guided preprocessing methods **do not improve** compression efficiency:

| Method | BD-Rate (Rotation) | BD-Rate (Zoom) |
|--------|-------------------|----------------|
| Bilateral | +18.3% | +18.0% |
| Blur | +31.4% | +53.8% |

**Positive BD-Rate means the method requires MORE bits for the same quality.** Preprocessing hurts efficiency rather than improving it.

## Project Structure

```
final/
├── config.py                 # Dataset paths and configuration
├── utils.py                  # Core utilities (PSNR, SSIM, etc.)
├── method2_depth_quality.py  # Depth-guided preprocessing (bilateral/blur)
├── experiment_proper.py      # Main experiment with BD-Rate calculation
├── generate_paper_figures.py # Generate publication figures
├── benchmark.py              # Quick benchmark script
├── report/
│   ├── paper.tex            # IEEE format paper
│   ├── paper.pdf            # Compiled paper
│   ├── fig1_rd_curves.png   # Rate-distortion curves
│   ├── fig2_bd_rate.png     # BD-Rate comparison
│   ├── fig3_quality_crf28.png
│   ├── fig4_tradeoff.png
│   └── fig5_ssim_curves.png
└── outputs/                  # Experiment results (JSON + videos)
```

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: numpy, opencv-python, pillow, matplotlib, scipy, scikit-image

FFmpeg must be installed with libx264 support.

## Running Experiments

```bash
# Run the full experiment (both segments, 6 CRF values)
python experiment_proper.py

# Generate publication figures
python generate_paper_figures.py
```

## Methodology

### BD-Rate (Bjontegaard Delta Rate)

BD-Rate is the standard metric in video compression research. It measures the average percentage bitrate difference between two methods at equivalent quality levels:

- **Negative BD-Rate**: Method saves bits (better efficiency)
- **Positive BD-Rate**: Method requires more bits (worse efficiency)

We use 6 CRF values (18, 23, 28, 33, 38, 43) to generate proper RD curves for fair comparison.

### Test Segments

- **Rotation** (frames 0-149): Camera rotates around Mandelbulb fractal. Stable depth.
- **Zoom** (frames 300-449): Camera zooms in. Dynamic depth - background becomes foreground.

## Why Preprocessing Fails

1. **Quality is measured against originals**: Preprocessing degrades the source, so decoded frames always differ more from originals.

2. **Modern encoders are already efficient**: H.264 efficiently encodes smooth regions with minimal bits. Preprocessing provides little additional benefit.

3. **Information is permanently lost**: Unlike encoder QP modulation, preprocessing destroys information before encoding.

## When Trade-offs May Be Acceptable

- Video conferencing (background quality unimportant)
- Extreme bandwidth constraints
- Aesthetic blur effects (bokeh)

These are **quality trade-offs**, not efficiency improvements.

## References

- Bjontegaard, G. "Calculation of average PSNR differences between RD-curves." ITU-T SG16 Doc. VCEG-M33, 2001.
- Tech, G. et al. "Overview of the Multiview and 3D Extensions of High Efficiency Video Coding." IEEE Trans. CSVT, 2016.
