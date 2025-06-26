#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练并立即评估 KLT 变换的脚本"""

import argparse
import pathlib
import math
import os
import subprocess
import csv
import numpy as np
from numpy.linalg import svd, norm


def video_to_residual_blocks(path, num_frames, blk, offset, width, height, skip=1):
    """从 YUV420 文件提取 Y 残差块，返回形状 (blk*blk, N)"""
    y_size = width * height
    uv_size = y_size // 4
    frame_sz = y_size + 2 * uv_size
    blocks = []
    with open(path, "rb") as f:
        f.seek(frame_sz * skip, os.SEEK_CUR)
        for _ in range(num_frames):
            data = f.read(frame_sz)
            if len(data) < frame_sz:
                break
            Y = (
                np.frombuffer(data, count=y_size, dtype=np.uint8)
                .reshape((height, width))
                .astype(np.float64)
                - offset
            )
            bh, bw = (height // blk) * blk, (width // blk) * blk
            B = (
                Y[:bh, :bw]
                .reshape(bh // blk, blk, bw // blk, blk)
                .swapaxes(1, 2)
                .reshape(-1, blk * blk)
                .T
            )
            blocks.append(B)
    if not blocks:
        raise RuntimeError(f"No valid frames in {path}")
    return np.hstack(blocks)


def quantize(A, q):
    return np.round(A / q).astype(int)


def run_cabac(coeff_file, out_bin, w, h, cabac_exe, cfg):
    data = np.loadtxt(coeff_file, dtype=int)
    if not np.any(data):
        return 0
    subprocess.run(
        [cabac_exe, "-c", cfg, "--coeffs", coeff_file, "--cabacBin", out_bin, "-wdt", str(w), "-hgt", str(h)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return os.path.getsize(out_bin) * 8 - 16


def train_klt(args):
    X_list = []
    for vid, w, h in zip(args.train_video, args.widths, args.heights):
        X_list.append(
            video_to_residual_blocks(
                vid,
                args.train_frames,
                args.block,
                args.residual_offset,
                w,
                h,
                args.skip_frames,
            )
        )
    X = np.hstack(X_list)
    if args.train_blocks:
        perm = np.random.permutation(X.shape[1])[: args.train_blocks]
        X = X[:, perm]
    U, _, _ = svd(X, full_matrices=False)
    return U.astype(np.float64)


def evaluate_klt(D, args):
    X = video_to_residual_blocks(
        args.test_video,
        args.test_frames,
        args.block,
        args.residual_offset,
        args.width,
        args.height,
        args.test_skip_frames,
    )
    coeffs = D.T @ X
    Ai = quantize(coeffs, args.quant)
    rec = D @ (Ai * args.quant)
    total_err = float(norm(X - rec) ** 2)
    pixels = args.test_frames * (args.width // args.block) * (args.height // args.block) * (args.block * args.block)
    mse = total_err / pixels
    psnr = 10 * math.log10((255 ** 2) / mse) if mse > 0 else float("inf")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coeff_path = out_dir / "coeffs.txt"
    np.savetxt(coeff_path, Ai.T, fmt="%d")
    bin_path = out_dir / "output.bin"
    bits = run_cabac(str(coeff_path), str(bin_path), args.width, args.height, args.cabac_exe, args.tu_cfg)
    bpp = bits / pixels
    return psnr, bpp, out_dir


def main():
    ap = argparse.ArgumentParser(description="KLT 训练后立即评估")
    ap.add_argument("--train_video", nargs="+", required=True)
    ap.add_argument("--widths", nargs="+", type=int, required=True)
    ap.add_argument("--heights", nargs="+", type=int, required=True)
    ap.add_argument("--train_frames", type=int, default=1)
    ap.add_argument("--skip_frames", type=int, default=1)
    ap.add_argument("--train_blocks", type=int, default=None)

    ap.add_argument("--test_video", required=True)
    ap.add_argument("--width", type=int, default=None,
                    help="测试序列宽度(默认与训练第一序列相同)")
    ap.add_argument("--height", type=int, default=None,
                    help="测试序列高度(默认与训练第一序列相同)")
    ap.add_argument("--test_frames", type=int, default=None,
                    help="测试帧数(默认与训练帧数相同)")
    ap.add_argument("--test_skip_frames", type=int, default=None,
                    help="测试起始跳过帧数(默认与训练相同)")

    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--residual_offset", type=int, default=128)
    ap.add_argument("--quant", type=int, default=32)
    ap.add_argument("--qp", type=int, required=True)

    ap.add_argument("--output_dir", default="klt_eval")
    ap.add_argument("--cabac_exe", default="CABAC.exe")
    ap.add_argument("--tu_cfg", default="TU8.cfg")
    args = ap.parse_args()

    if not (len(args.train_video) == len(args.widths) == len(args.heights)):
        raise ValueError("训练视频、widths、heights 数量必须一致")

    if args.width is None:
        args.width = args.widths[0]
    if args.height is None:
        args.height = args.heights[0]
    if args.test_frames is None:
        args.test_frames = args.train_frames
    if args.test_skip_frames is None:
        args.test_skip_frames = args.skip_frames

    print("Training KLT ...")
    D = train_klt(args)

    print("Evaluating KLT ...")
    psnr, bpp, out_dir = evaluate_klt(D, args)

    print(f"PSNR: {psnr:.3f} dB")
    print(f"BPP : {bpp:.6f}")

    csv_path = out_dir / "summary.csv"
    header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["Scheme", "QP", "PSNR(dB)", "bpp"])
        w.writerow(["KLT", args.qp, f"{psnr:.3f}", f"{bpp:.6f}"])

    print("Results saved to", csv_path)


if __name__ == "__main__":
    main()