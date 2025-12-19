"""
Test at VERY low bitrates where baseline might struggle more.
Maybe preprocessing helps when bandwidth is extremely constrained.
"""

from perceptual_evaluation import run_perceptual_comparison

# Test at very small target sizes
for target_mb in [0.5, 1.0, 1.5]:
    print(f"\n{'='*70}")
    print(f"TARGET SIZE: {target_mb} MB")
    print(f"{'='*70}")
    results = run_perceptual_comparison(
        segment="rotation",
        num_frames=150,
        target_size_mb=target_mb
    )
