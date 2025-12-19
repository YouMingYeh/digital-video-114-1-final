"""
Depth-Aware Video Compression Research Pipeline
-----------------------------------------------
Goal: Prove that using depth maps to guide bit allocation (QP) improves 
      perceptual quality of foreground objects compared to standard encoding.

Steps:
1. Data Prep: Composite Mandelbulb frames over a complex (noisy) background.
   - Why? To stress the encoder. If background is black, compression is too easy.
2. Analysis: Calculate 'Importance Score' per frame based on Depth Map.
   - Logic: Closer/Larger objects = Lower QP (Better Quality).
3. Encoding:
   - Baseline: Standard H.264 (CRF).
   - Depth-Aware: H.264 with custom QP File (Frame-level bit allocation).
4. Evaluation:
   - Calculate PSNR of the FOREGROUND object only.
   - Compare File Size vs Foreground Quality.
"""

import cv2
import numpy as np
import os
import glob
import subprocess
import matplotlib.pyplot as plt
import shutil

# --- Configuration ---
DATASET_ROOT = "Mandelbulb_Dataset"
COLOR_DIR = os.path.join(DATASET_ROOT, "output_color")
DEPTH_DIR = os.path.join(DATASET_ROOT, "output_depth")
WORK_DIR = "research_workspace"
COMPOSITE_DIR = os.path.join(WORK_DIR, "composite_frames")
RESULTS_DIR = os.path.join(WORK_DIR, "results")

FPS = 30
DEPTH_THRESHOLD = 20  # Pixel value > 20 is considered "Foreground"

def ensure_dirs():
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(COMPOSITE_DIR)
    os.makedirs(RESULTS_DIR)

# --- Step 1: Dataset Preparation ---
def prepare_dataset():
    print("Step 1: Using Original Dataset (No Composite)...")
    # We just verify the files exist
    color_files = sorted(glob.glob(os.path.join(COLOR_DIR, "frame_*.png")))
    if not color_files:
        raise FileNotFoundError("No source images found!")
    return COLOR_DIR

# --- Step 2: Depth Analysis & QP File Generation ---
def generate_qp_file(base_qp):
    print(f"Step 2: Generating QP File (Base QP={base_qp})...")
    depth_files = sorted(glob.glob(os.path.join(DEPTH_DIR, "depth_*.png")))
    qp_file_path = os.path.join(RESULTS_DIR, "depth_aware.qp")
    
    with open(qp_file_path, 'w') as f:
        for i, d_path in enumerate(depth_files):
            depth = cv2.imread(d_path, cv2.IMREAD_GRAYSCALE)
            
            # Metric: Foreground Ratio (How close/large is the object?)
            fg_ratio = np.sum(depth > DEPTH_THRESHOLD) / depth.size
            
            # AGGRESSIVE Temporal Bit Allocation (Balanced)
            if fg_ratio > 0.40:
                # Super Close: Boost Quality
                qp = max(0, base_qp - 6) 
            elif fg_ratio > 0.20:
                # Medium: Save a little bit
                qp = min(51, base_qp + 2)
            else:
                # Far away: Save a lot
                qp = min(51, base_qp + 12)
                
            # FrameType: I for first, P for rest
            ftype = 'I' if i == 0 else 'P'
            f.write(f"{i} {ftype} {qp}\n")
            
    return qp_file_path

# --- Step 3: Encoding ---
def run_ffmpeg(cmd):
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def encode_videos(base_qp):
    print("Step 3: Encoding Videos...")
    # Use the original color directory directly
    input_pattern = os.path.join(COLOR_DIR, "frame_%03d.png")
    
    # 1. Reference (Lossless)
    ref_path = os.path.join(RESULTS_DIR, "reference.mp4")
    if not os.path.exists(ref_path):
        run_ffmpeg([
            'ffmpeg', '-y', '-framerate', str(FPS), '-i', input_pattern,
            '-c:v', 'libx264', '-qp', '0', '-preset', 'ultrafast', ref_path
        ])

    # 2. Baseline (Standard Constant QP)
    # We use CQP (Constant QP) for baseline to have a fair comparison with our CQP-based QPFile
    base_path = os.path.join(RESULTS_DIR, "baseline.mp4")
    run_ffmpeg([
        'ffmpeg', '-y', '-framerate', str(FPS), '-i', input_pattern,
        '-c:v', 'libx264', '-qp', str(base_qp), '-preset', 'medium', base_path
    ])
    
    # 3. Depth-Aware (QP File)
    qp_path = generate_qp_file(base_qp)
    aware_path = os.path.join(RESULTS_DIR, "depth_aware.mp4")
    run_ffmpeg([
        'ffmpeg', '-y', '-framerate', str(FPS), '-i', input_pattern,
        '-c:v', 'libx264', '-preset', 'medium',
        '-x264-params', f'qpfile={qp_path}', aware_path
    ])
    
    return ref_path, base_path, aware_path

# --- Step 4: Evaluation ---
def calculate_masked_psnr(ref_video, test_video):
    cap_ref = cv2.VideoCapture(ref_video)
    cap_test = cv2.VideoCapture(test_video)
    depth_files = sorted(glob.glob(os.path.join(DEPTH_DIR, "depth_*.png")))
    
    psnrs = []
    
    for i, d_path in enumerate(depth_files):
        ret1, frame_ref = cap_ref.read()
        ret2, frame_test = cap_test.read()
        if not ret1 or not ret2: break
        
        depth = cv2.imread(d_path, cv2.IMREAD_GRAYSCALE)
        mask = depth > DEPTH_THRESHOLD
        
        if np.sum(mask) == 0: continue # No foreground
        
        # Extract foreground pixels
        fg_ref = frame_ref[mask]
        fg_test = frame_test[mask]
        
        mse = np.mean((fg_ref - fg_test) ** 2)
        if mse == 0:
            psnr = 100
        else:
            psnr = 10 * np.log10(255**2 / mse)
        psnrs.append(psnr)
        
    return np.mean(psnrs)

def run_full_analysis():
    print("Step 5: Running Full RD Analysis (Multiple Bitrates)...")
    
    # Test multiple base QPs to generate a curve
    base_qps = [20, 25, 30, 35, 40]
    
    results = {
        "Baseline": {"size": [], "psnr": []},
        "DepthAware": {"size": [], "psnr": []}
    }
    
    # Ensure reference exists
    input_pattern = os.path.join(COLOR_DIR, "frame_%03d.png")
    ref_path = os.path.join(RESULTS_DIR, "reference.mp4")
    if not os.path.exists(ref_path):
        run_ffmpeg(['ffmpeg', '-y', '-framerate', str(FPS), '-i', input_pattern, '-c:v', 'libx264', '-qp', '0', '-preset', 'ultrafast', ref_path])

    for qp in base_qps:
        print(f"\nTesting Base QP: {qp}")
        
        # Baseline
        base_out = os.path.join(RESULTS_DIR, f"baseline_qp{qp}.mp4")
        run_ffmpeg(['ffmpeg', '-y', '-framerate', str(FPS), '-i', input_pattern, '-c:v', 'libx264', '-qp', str(qp), '-preset', 'medium', base_out])
        
        # Depth-Aware
        qp_file = generate_qp_file(qp)
        aware_out = os.path.join(RESULTS_DIR, f"aware_qp{qp}.mp4")
        run_ffmpeg(['ffmpeg', '-y', '-framerate', str(FPS), '-i', input_pattern, '-c:v', 'libx264', '-preset', 'medium', '-x264-params', f'qpfile={qp_file}', aware_out])
        
        # Metrics
        s_base = os.path.getsize(base_out) / 1024 / 1024
        p_base = calculate_masked_psnr(ref_path, base_out)
        
        s_aware = os.path.getsize(aware_out) / 1024 / 1024
        p_aware = calculate_masked_psnr(ref_path, aware_out)
        
        results["Baseline"]["size"].append(s_base)
        results["Baseline"]["psnr"].append(p_base)
        results["DepthAware"]["size"].append(s_aware)
        results["DepthAware"]["psnr"].append(p_aware)
        
        print(f"  Base: {s_base:.2f}MB, {p_base:.2f}dB | Aware: {s_aware:.2f}MB, {p_aware:.2f}dB")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(results["Baseline"]["size"], results["Baseline"]["psnr"], 'o-', label='Baseline (Standard)')
    plt.plot(results["DepthAware"]["size"], results["DepthAware"]["psnr"], 's-', label='Depth-Aware (Ours)')
    plt.xlabel('File Size (MB)')
    plt.ylabel('ROI PSNR (dB)')
    plt.title('Rate-Distortion Curve: Depth-Aware Efficiency')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "rd_curve_final.png"))
    print(f"\nSaved RD Curve to {os.path.join(RESULTS_DIR, 'rd_curve_final.png')}")

if __name__ == "__main__":
    ensure_dirs()
    prepare_dataset()
    run_full_analysis()
