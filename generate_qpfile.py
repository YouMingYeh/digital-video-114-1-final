"""
QP File Generator (Debug & Verify)
----------------------------------
Generates a QP file for x264 based on Depth Maps.
Strictly follows: final_qp = base_qp - offset.
"""

import cv2
import numpy as np
import os
import glob
import sys

def generate_qp_file(depth_dir, output_path, base_qp, depth_threshold=20):
    print(f"\n--- Generating QP File: {output_path} ---")
    print(f"Base QP: {base_qp}")
    
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "depth_*.png")))
    if not depth_files:
        raise FileNotFoundError(f"No depth maps found in {depth_dir}")

    with open(output_path, 'w') as f:
        for i, d_path in enumerate(depth_files):
            # 1. Read Actual Pixel Data
            depth = cv2.imread(d_path, cv2.IMREAD_GRAYSCALE)
            if depth is None:
                raise ValueError(f"⚠️ Depth Map Read Error: Could not read {d_path}")
            
            # 2. Calculate Statistics
            avg_brightness = np.mean(depth)
            fg_ratio = np.sum(depth > depth_threshold) / depth.size
            
            # Self-Check
            if avg_brightness == 0 and i < 10: # Check first few frames
                print(f"⚠️ Warning: Frame {i} is completely black (Avg Depth: 0). Is this expected?")

            # 3. Logic: final_qp = base_qp - offset
            # Strategy:
            # - Close (High FG Ratio): High Quality (Negative Offset)
            # - Far (Low FG Ratio): Low Quality (Positive Offset)
            
            offset = 0
            if fg_ratio > 0.40:
                offset = -6  # Boost quality (QP 30 -> 24)
            elif fg_ratio > 0.20:
                offset = 2   # Slight save (QP 30 -> 32)
            else:
                offset = 12  # Crush background (QP 30 -> 42)
            
            final_qp = int(np.clip(base_qp + offset, 0, 51))
            
            # 4. Frame Type
            ftype = 'I' if i == 0 else 'P'
            
            # 5. Write to file
            f.write(f"{i} {ftype} {final_qp}\n")
            
            # 6. Proof of Life Logging (First 5 frames + every 50th)
            if i < 5 or i % 50 == 0:
                print(f"Frame {i:03d} | AvgDepth: {avg_brightness:6.2f} | FG: {fg_ratio*100:5.1f}% | BaseQP: {base_qp} -> FinalQP: {final_qp} (Offset: {offset})")

    print(f"✅ QP File generated: {output_path}")
    return output_path

if __name__ == "__main__":
    # Test Run
    DEPTH_DIR = "Mandelbulb_Dataset/output_depth"
    RESULTS_DIR = "research_workspace/results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Test with Base QP 30
    generate_qp_file(DEPTH_DIR, os.path.join(RESULTS_DIR, "debug_qp30.qp"), base_qp=30)
    
    # Test with Base QP 20 (Should see different Final QPs)
    generate_qp_file(DEPTH_DIR, os.path.join(RESULTS_DIR, "debug_qp20.qp"), base_qp=20)
