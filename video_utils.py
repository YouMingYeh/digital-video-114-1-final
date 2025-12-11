#!/usr/bin/env python3
"""
Video Utilities Module
Common FFmpeg operations for video analysis and processing.
"""

import subprocess
import json
import os
import re
import statistics


def get_video_info(video_path):
    """Extract video metadata using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout) if result.returncode == 0 else {}


def _parse_psnr_from_stderr(stderr):
    """Extract PSNR value from FFmpeg stderr output."""
    for line in stderr.split('\n'):
        if 'average:' in line.lower():
            match = re.search(r'average:(\d+\.?\d*)', line)
            if match:
                return float(match.group(1))
    return 0.0


def _parse_ssim_from_stderr(stderr):
    """Extract SSIM value from FFmpeg stderr output."""
    for line in stderr.split('\n'):
        if 'All:' in line:
            match = re.search(r'All:(\d+\.?\d*)', line)
            if match:
                return float(match.group(1))
    return 0.0


def calculate_psnr(reference, distorted, crop_filter=None):
    """
    Calculate PSNR between two videos.
    
    Args:
        reference: Path to reference video
        distorted: Path to distorted video
        crop_filter: Optional crop filter (e.g., "crop=1024:1024:512:512")
    
    Returns:
        float: PSNR value in dB, or 0.0 on failure
    """
    if crop_filter:
        filter_chain = f'[0:v]{crop_filter}[a];[1:v]{crop_filter}[b];[a][b]psnr'
        cmd = ['ffmpeg', '-i', distorted, '-i', reference, 
               '-filter_complex', filter_chain, '-f', 'null', '-']
    else:
        cmd = ['ffmpeg', '-y', '-i', distorted, '-i', reference, 
               '-lavfi', 'psnr', '-f', 'null', '-']
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return _parse_psnr_from_stderr(result.stderr)


def calculate_ssim(reference, distorted):
    """Calculate SSIM between two videos."""
    cmd = ['ffmpeg', '-y', '-i', distorted, '-i', reference, 
           '-lavfi', 'ssim', '-f', 'null', '-']
    result = subprocess.run(cmd, capture_output=True, text=True)
    return _parse_ssim_from_stderr(result.stderr)


def extract_frame(video_path, frame_num, output_path):
    """Extract a specific frame from video as PNG."""
    cmd = [
        'ffmpeg', '-y', '-v', 'quiet',
        '-i', video_path,
        '-vf', f'select=eq(n\\,{frame_num})',
        '-vframes', '1',
        output_path
    ]
    subprocess.run(cmd, capture_output=True)
    return output_path if os.path.exists(output_path) else None


def create_comparison_image(left_frame, right_frame, output_path):
    """Create side-by-side comparison image."""
    cmd = [
        'ffmpeg', '-y', '-v', 'quiet',
        '-i', left_frame, '-i', right_frame,
        '-filter_complex', 'hstack=inputs=2',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    return output_path if result.returncode == 0 else None


def analyze_bitrate_distribution(video_path):
    """Analyze frame-by-frame bitrate distribution."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'packet=size',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    sizes = [int(s) for s in result.stdout.strip().split('\n') if s.isdigit()]
    
    if len(sizes) > 1:
        return {
            'avg': statistics.mean(sizes),
            'stdev': statistics.stdev(sizes),
            'min': min(sizes),
            'max': max(sizes)
        }
    return None


def create_reference_video(source_pattern, output_path, fps=30):
    """Create a lossless reference video from source images."""
    if os.path.exists(output_path):
        return output_path
    
    cmd = [
        'ffmpeg', '-y', '-v', 'quiet',
        '-framerate', str(fps),
        '-i', source_pattern,
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    return output_path if result.returncode == 0 else None


def get_file_size_mb(path):
    """Get file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0
