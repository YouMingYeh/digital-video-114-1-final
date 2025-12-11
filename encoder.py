#!/usr/bin/env python3
"""
Video Encoder - H.264 encoding with ROI optimization via Adaptive Quantization.
"""

import os
import subprocess
import argparse


class VideoEncoder:
    """Handles baseline and ROI-optimized video encoding."""

    def __init__(self, color_dir, output_dir=".", fps=30):
        self.color_dir = color_dir
        self.output_dir = output_dir
        self.fps = fps
        os.makedirs(output_dir, exist_ok=True)

    def _run_ffmpeg(self, cmd, description):
        """Execute FFmpeg command with status logging."""
        print(f"🎬 {description}...")
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("✅ Done.")
            return True
        print(f"❌ Failed with code {result.returncode}")
        return False

    def _build_cmd(self, output_name, crf=None, bitrate=None, x264_params=None, two_pass=False, pass_num=None):
        """Build FFmpeg command with common options."""
        output_path = os.path.join(self.output_dir, output_name)
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(self.fps),
            '-i', f'{self.color_dir}/frame_%03d.png',
            '-c:v', 'libx264',
            '-preset', 'slow',
        ]
        if crf is not None:
            cmd.extend(['-crf', str(crf)])
        if bitrate is not None:
            cmd.extend(['-b:v', f'{bitrate}k'])
        if x264_params:
            cmd.extend(['-x264-params', x264_params])
        if two_pass:
            passlog = os.path.join(self.output_dir, "ffmpeg2pass")
            cmd.extend(['-pass', str(pass_num), '-passlogfile', passlog])
            if pass_num == 1:
                cmd.extend(['-f', 'null', '-'])
                return cmd, None
        cmd.extend(['-pix_fmt', 'yuv420p', output_path])
        return cmd, output_path

    def encode_baseline(self, crf=23):
        """Standard H.264 encoding without ROI optimization."""
        cmd, _ = self._build_cmd("baseline.mp4", crf=crf)
        return self._run_ffmpeg(cmd, f"Baseline (CRF {crf})")

    def encode_roi(self, crf=23, aq_strength=2.0):
        """ROI-optimized encoding using Adaptive Quantization."""
        params = f'aq-mode=3:aq-strength={aq_strength}:psy-rd=1.0:deblock=-1,-1'
        cmd, _ = self._build_cmd("optimized.mp4", crf=crf, x264_params=params)
        return self._run_ffmpeg(cmd, f"ROI (CRF {crf}, AQ {aq_strength})")

    def encode_aggressive(self, crf=28, aq_strength=3.0):
        """Aggressive size reduction with ROI preservation."""
        params = f'aq-mode=3:aq-strength={aq_strength}:qcomp=0.8:psy-rd=1.0'
        cmd, _ = self._build_cmd("aggressive_roi.mp4", crf=crf, x264_params=params)
        return self._run_ffmpeg(cmd, f"Aggressive (CRF {crf}, AQ {aq_strength})")

    def encode_matched(self, target_size_mb):
        """Two-pass encoding to match a target file size."""
        num_frames = len([f for f in os.listdir(self.color_dir) if f.endswith('.png')])
        duration = num_frames / self.fps
        bitrate = int((target_size_mb * 8 * 1024) / duration)
        
        print(f"📊 Target: {target_size_mb:.2f} MB | Bitrate: {bitrate}k")
        
        params = 'aq-mode=3:aq-strength=2.0:psy-rd=1.0'
        
        # Pass 1
        cmd1, _ = self._build_cmd("optimized_matched.mp4", bitrate=bitrate, 
                                   x264_params=params, two_pass=True, pass_num=1)
        if not self._run_ffmpeg(cmd1, "Pass 1"):
            return False
        
        # Pass 2
        cmd2, _ = self._build_cmd("optimized_matched.mp4", bitrate=bitrate,
                                   x264_params=params, two_pass=True, pass_num=2)
        success = self._run_ffmpeg(cmd2, "Pass 2")
        
        # Cleanup pass logs
        for ext in ['', '.log', '.log.mbtree', '.log.temp']:
            path = os.path.join(self.output_dir, f"ffmpeg2pass{ext}")
            if os.path.exists(path):
                os.remove(path)
        return success

    def encode_starved(self, bitrate_kbps=500):
        """Low-bitrate comparison to demonstrate ROI benefit."""
        print(f"📉 Starved Bitrate Demo ({bitrate_kbps} kbps)...")
        
        # Baseline starved
        cmd1, _ = self._build_cmd("baseline_starved.mp4", bitrate=bitrate_kbps)
        self._run_ffmpeg(cmd1, "Starved Baseline")
        
        # ROI starved
        params = 'aq-mode=3:aq-strength=4.0:psy-rd=1.0'
        cmd2, _ = self._build_cmd("roi_starved.mp4", bitrate=bitrate_kbps, x264_params=params)
        self._run_ffmpeg(cmd2, "Starved ROI")

def main():
    parser = argparse.ArgumentParser(description="Video Encoder with ROI Optimization")
    parser.add_argument('mode', choices=['baseline', 'roi', 'aggressive', 'matched', 'starved', 'all'])
    parser.add_argument('--color_dir', default="./Mandelbulb_Dataset/output_color")
    parser.add_argument('--output_dir', default=".")
    parser.add_argument('--target_mb', type=float, default=30.38)
    parser.add_argument('--bitrate', type=int, default=500)

    args = parser.parse_args()
    encoder = VideoEncoder(args.color_dir, args.output_dir)

    if args.mode in ('baseline', 'all'):
        encoder.encode_baseline()
    if args.mode in ('roi', 'all'):
        encoder.encode_roi()
    if args.mode in ('aggressive', 'all'):
        encoder.encode_aggressive()
    if args.mode in ('matched', 'all'):
        encoder.encode_matched(args.target_mb)
    if args.mode in ('starved', 'all'):
        encoder.encode_starved(args.bitrate)


if __name__ == "__main__":
    main()
