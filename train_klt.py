#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本：基于残差块计算 KLT 字典
"""
import os
import argparse
import pathlib
import numpy as np
from numpy.linalg import svd


def video_to_residual_blocks(path, num_frames, blk, offset, width, height, skip=1):
    """从 4:2:0 YUV420 文件提取 Y 残差块，返回形状 (blk*blk, N)"""
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
            Y = (np.frombuffer(data, count=y_size, dtype=np.uint8)
                   .reshape((height, width)).astype(np.float64) - offset)
            bh, bw = (height // blk) * blk, (width // blk) * blk
            B = (Y[:bh, :bw]
                 .reshape(bh // blk, blk, bw // blk, blk)
                 .swapaxes(1, 2)
                 .reshape(-1, blk * blk).T)
            blocks.append(B)
    if not blocks:
        raise RuntimeError(f"No valid frames in {path}")
    return np.hstack(blocks)


def main():
    ap = argparse.ArgumentParser(description="训练 KLT 字典")
    ap.add_argument("--train_video", nargs="+", required=True,
                    help="训练用视频序列（可多个）")
    ap.add_argument("--train_frames", type=int, default=1)
    ap.add_argument("--skip_frames", type=int, default=1)
    ap.add_argument("--widths", nargs="+", type=int, required=True,
                    help="每个视频的宽度（顺序对应）")
    ap.add_argument("--heights", nargs="+", type=int, required=True,
                    help="每个视频的高度（顺序对应）")
    ap.add_argument("--residual_offset", type=int, default=128)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--train_blocks", type=int, default=None,
                    help="随机抽取块数量（可选）")
    ap.add_argument("--output", default="trained_dicts/KLT.npy")
    args = ap.parse_args()

    if not (len(args.train_video) == len(args.widths) == len(args.heights)):
        raise ValueError("train_video、widths、heights数量必须一致！")

    print("Extracting training residual blocks …")
    X_list = []
    for vid, w, h in zip(args.train_video, args.widths, args.heights):
        X_list.append(
            video_to_residual_blocks(
                vid, args.train_frames, args.block,
                args.residual_offset, w, h, args.skip_frames
            )
        )
    X = np.hstack(X_list)
    if args.train_blocks:
        perm = np.random.permutation(X.shape[1])[: args.train_blocks]
        X = X[:, perm]

    print("Computing KLT matrix …")
    U, _, _ = svd(X, full_matrices=False)
    D = U.astype(np.float64)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), D)
    print("KLT dictionary saved to", out_path)


if __name__ == "__main__":
    main()