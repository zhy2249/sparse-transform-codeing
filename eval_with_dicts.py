#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本：读取已有字典 + DCT/MTS 进行五种方案评估
"""
import os
import argparse
import pathlib
import math
import csv
import subprocess
from CABAC_Estimator import CabacEstimator

import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import orthogonal_mp_gram
from tqdm import tqdm

# ------------------------------------------------------------
# 提取残差块
# ------------------------------------------------------------
def video_to_residual_blocks(path, num_frames, blk, offset, w, h, skip=1):
    y_size  = w * h
    uv_size = y_size // 4
    frame_sz= y_size + 2*uv_size
    blocks=[]
    with open(path, "rb") as f:
        f.seek(frame_sz*skip, os.SEEK_CUR)
        for _ in range(num_frames):
            data=f.read(frame_sz)
            if len(data)<frame_sz: break
            Y=(np.frombuffer(data, count=y_size, dtype=np.uint8)
                 .reshape(h,w).astype(np.float64)-offset)
            bh,bw=(h//blk)*blk,(w//blk)*blk
            B=(Y[:bh,:bw]
               .reshape(bh//blk,blk,bw//blk,blk)
               .swapaxes(1,2)
               .reshape(-1,blk*blk).T)
            blocks.append(B)
    if not blocks:
        raise RuntimeError(f"No frames in {path}")
    return np.hstack(blocks)

# ------------------------------------------------------------
# 外部 CABAC 调用
# ------------------------------------------------------------
def run_cabac(coeff_file, out_bin, w, h, cabac_exe, cfg):
    data=np.loadtxt(coeff_file, dtype=int)
    if not np.any(data): return 0
    subprocess.run(
        [cabac_exe,"-c",cfg,
         "--coeffs",coeff_file,
         "--cabacBin",out_bin,
         "-wdt",str(w),"-hgt",str(h)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    # 去掉文件头 16 bit
    return os.path.getsize(out_bin)*8-16

# ------------------------------------------------------------
# 量化
# ------------------------------------------------------------
def omp(D, X, k):
    G  = D.T @ D
    DX = D.T @ X
    return orthogonal_mp_gram(G, DX, n_nonzero_coefs=k).astype(np.float64)

def quantize(A, q):
    return np.round(A/q).astype(int)

# ------------------------------------------------------------
# 读取 .npy 字典列表
# ------------------------------------------------------------
def load_dicts(dict_dir, prefix):
    p=pathlib.Path(dict_dir)
    files=sorted(p.glob(f"{prefix}*.npy"))
    return [np.load(str(f)) for f in files]

# ------------------------------------------------------------
# DCT / DST7 / DCT8 1D 矩阵
# ------------------------------------------------------------
def dct2_1d(N):
    T=np.zeros((N,N))
    for k in range(N):
        for n in range(N):
            T[k,n]=math.cos(math.pi*(2*n+1)*k/(2*N))
    T[0,:]*=1/math.sqrt(2)
    return T*math.sqrt(2/N)

def dst7_1d(N):
    T=np.zeros((N,N))
    for k in range(N):
        for n in range(N):
            T[k,n]=math.sin(math.pi*(2*n+1)*(k+1)/(2*N+1))
    return T*math.sqrt(2/(2*N+1))

def dct8_1d(N):
    T=np.zeros((N,N))
    for k in range(N):
        for n in range(N):
            T[k,n]=math.cos(math.pi*(2*n+1)*(2*k+1)/(4*N))
    return T*math.sqrt(2/N)

# ------------------------------------------------------------
# 生成 VVC MTS 五种 2D 字典
# ------------------------------------------------------------
def mts_2d_dicts(N):
    T2=dct2_1d(N); T7=dst7_1d(N); T8=dct8_1d(N)
    combos=[
        ("DCT2xDCT2",T2,T2),
        ("DST7xDST7",T7,T7),
        ("DCT8xDCT8",T8,T8),
        ("DST7xDCT8",T7,T8),
        ("DCT8xDST7",T8,T7),
    ]
    tags=[]; dicts=[]
    for name,Tv,Th in combos:
        D=np.zeros((N*N,N*N))
        idx=0
        for u in range(N):
            for v in range(N):
                basis=np.outer(Tv[u,:],Th[v,:])
                D[:,idx]=basis.flatten(); idx+=1
        D/= (norm(D,axis=0,keepdims=True)+1e-12)
        tags.append(name); dicts.append(D)
    return tags, dicts

# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--test_video",required=True)
    ap.add_argument("--test_frames",type=int,default=10)
    ap.add_argument("--skip_frames",type=int,default=1)
    ap.add_argument("--width", type=int,required=True)
    ap.add_argument("--height",type=int,required=True)
    ap.add_argument("--residual_offset",type=int,default=128)
    ap.add_argument("--block",type=int,default=8)
    ap.add_argument("--quant",type=int,default=32)
    ap.add_argument("--qp", type=int,required=True)
    ap.add_argument("--sparsity", type=int, default=3, help="OMP 稀疏度")
    ap.add_argument("--lam_scale",type=float,default=1.0)
    ap.add_argument("--lam_KSVD", type=float, default=100.0, help="外部指定的 KSVD部分λ ")
    ap.add_argument("--dict_dir", required=True,
                    help="训练字典(.npy)所在目录")
    ap.add_argument("--output_dir",default="eval_results")
    ap.add_argument("--cabac_exe",default="CABAC.exe")
    ap.add_argument("--tu_cfg",default="TU8.cfg")
    ap.add_argument("--skip_schemes",nargs="*",default=[],
                    choices=["DCT","MTS","KSVD","MultiKSVD","IterMultiKSVD"])
    args=ap.parse_args()

    # 1) 准备
    out_root=pathlib.Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    blk=args.block; w=args.width; h=args.height
    X_test=video_to_residual_blocks(
        args.test_video,args.test_frames,blk,
        args.residual_offset,w,h,args.skip_frames
    )
    Nblocks=X_test.shape[1]
    total_pixels=args.test_frames*(w//blk)*(h//blk)*(blk*blk)

    # λ 计算
    lam_base=0.85*2**((args.qp-12)/3)
    lam=lam_base*args.lam_scale

    est = CabacEstimator()
    results=[]

    # ---------- DCT 单字典（按帧处理） ----------
    if "DCT" not in args.skip_schemes:
        print("Evaluating DCT …")
        # 预先构造 DCT 字典
        D_dct = dct2_1d(blk)
        D2 = np.zeros((blk * blk, blk * blk))
        idx = 0
        for u in range(blk):
            for v in range(blk):
                D2[:, idx] = np.outer(D_dct[u, :], D_dct[v, :]).flatten()
                idx += 1

        # 全局累加（可选，看你是否要总结果）
        total_bits = 0
        total_err = 0.0

        # 对每一帧单独处理
        for frame_idx in range(args.test_frames):
            # 读取并只取当前这一帧的残差块
            X_frame = video_to_residual_blocks(
                args.test_video, 1, blk,
                args.residual_offset, w, h,
                skip=args.skip_frames + frame_idx
            )
            Nblocks_f = X_frame.shape[1]
            sum_err_f = 0.0
            coeffs_f = []

            # 对当前帧所有块做 DCT+量化
            for i in range(Nblocks_f):
                x = X_frame[:, i:i + 1]
                Ar = D2.T @ x
                Ai = quantize(Ar, args.quant)
                rec = D2 @ (Ai * args.quant)
                err = float(norm(x - rec.reshape(x.shape))) ** 2
                sum_err_f += err
                coeffs_f.append(Ai.flatten().tolist())

            # 写这一帧的系数文件
            out_dir = out_root / f"QP={args.qp}" / "DCT" / f"frame_{frame_idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            coeff_path = out_dir / f"coeffs_{frame_idx}.txt"
            with open(coeff_path, "w") as f:
                for vec in coeffs_f:
                    f.write(" ".join(str(int(v)) for v in vec) + "\n")

            # 调用 CABAC，写出并累计比特
            bin_path = out_dir / f"out_{frame_idx}.bin"
            bits_f = run_cabac(str(coeff_path), str(bin_path),
                               w, h, args.cabac_exe, args.tu_cfg)

            # 计算这一帧的指标
            pixels_f = Nblocks_f * blk * blk
            mse_f = sum_err_f / pixels_f
            psnr_f = 10 * math.log10((255 ** 2) / mse_f) if mse_f > 0 else float('inf')
            bpp_f = bits_f / pixels_f

            # 累加全局（可选）
            total_bits += bits_f
            total_err += sum_err_f

            # 存到 results，如果你想每帧一行
            results.append([f"DCT_f{frame_idx}", args.qp,
                            f"{psnr_f:.3f}", f"{bpp_f:.6f}"])


        total_pixels = sum(args.block*args.block * (w//blk)*(h//blk) for _ in range(args.test_frames))
        mse_all   = total_err  / total_pixels
        psnr_all  = 10 * math.log10((255**2) / mse_all)
        bpp_all   = total_bits / total_pixels
        results.append(["DCT_overall", args.qp,
                        f"{psnr_all:.3f}", f"{bpp_all:.6f}"])

    # ---------- MTS 多字典（按帧处理 & 记录全帧信息） ----------
    if "MTS" not in args.skip_schemes:
        print("Evaluating MTS …")
        tags_mts, D_list_mts = mts_2d_dicts(blk)

        num_mts = len(D_list_mts)
        idx_bits_mts = math.ceil(math.log2(num_mts))

        total_bits = 0
        total_err = 0.0

        # 按帧循环
        for frame_idx in range(args.test_frames):
            # 只取这一帧的残差块
            X_frame = video_to_residual_blocks(
                args.test_video, 1, blk,
                args.residual_offset, w, h,
                skip=args.skip_frames + frame_idx
            )
            Nblocks_f = X_frame.shape[1]
            sum_err_f = 0.0
            coeffs_f = []

            # 对当前帧的每个块做 MTS+RD 选择
            for i in range(Nblocks_f):
                x = X_frame[:, i:i + 1]
                best_J = None
                best_Ai = None
                best_err = 0.0

                for D in D_list_mts:
                    Ar = D.T @ x
                    Ai = quantize(Ar, args.quant)
                    rec = D @ (Ai * args.quant)
                    err = float(norm(x - rec.reshape(x.shape))) ** 2

                    bits = est.estimate_block_bits(
                        Ai.reshape((blk, blk)), update=False
                    )
                    bits_with_idx = bits + (0 if np.all(Ai == 0) else idx_bits_mts)

                    J = err + lam * bits_with_idx
                    if best_J is None or J < best_J:
                        best_J = J
                        best_Ai = Ai.copy()
                        best_err = err

                est.estimate_block_bits(best_Ai.reshape((blk, blk)), update=True)
                sum_err_f += best_err
                coeffs_f.append(best_Ai.flatten().tolist())

            # 写这一帧的系数文件
            out_dir = out_root / f"QP={args.qp}" / "MTS" / f"frame_{frame_idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            coeff_path = out_dir / f"coeffs_{frame_idx}.txt"
            with open(coeff_path, "w") as f:
                for vec in coeffs_f:
                    f.write(" ".join(str(int(v)) for v in vec) + "\n")

            # 调用 CABAC 并统计比特
            bin_path = out_dir / f"out_{frame_idx}.bin"
            bits_frame = run_cabac(
                str(coeff_path), str(bin_path),
                w, h, args.cabac_exe, args.tu_cfg
            )
            nonzero_blocks = sum(1 for vec in coeffs_f if any(v != 0 for v in vec))
            bits_frame += nonzero_blocks * idx_bits_mts

            # 累加全局
            total_bits += bits_frame
            total_err += sum_err_f

            # 这一帧的指标
            pixels_f = Nblocks_f * blk * blk
            mse_f = sum_err_f / pixels_f
            psnr_f = 10 * math.log10((255 ** 2) / mse_f) if mse_f > 0 else float('inf')
            bpp_f = bits_frame / pixels_f
            results.append([f"MTS_f{frame_idx}", args.qp,
                            f"{psnr_f:.3f}", f"{bpp_f:.6f}"])

        # 全帧汇总指标
        total_pixels = args.test_frames * (w // blk) * (h // blk) * (blk * blk)
        mse_all = total_err / total_pixels
        psnr_all = 10 * math.log10((255 ** 2) / mse_all) if mse_all > 0 else float('inf')
        bpp_all = total_bits / total_pixels
        results.append(["MTS_overall", args.qp,
                        f"{psnr_all:.3f}", f"{bpp_all:.6f}"])

    # ---------- KSVD 单字典（按帧处理 & 记录全帧信息） ----------
    if "KSVD" not in args.skip_schemes:
        print("Evaluating KSVD …")
        Dk = np.load(str(pathlib.Path(args.dict_dir) / "KSVD.npy"))
        total_bits = 0
        total_err = 0.0

        for frame_idx in range(args.test_frames):
            # 只取当前帧的残差块
            X_frame = video_to_residual_blocks(
                args.test_video, 1, blk,
                args.residual_offset, w, h,
                skip=args.skip_frames + frame_idx
            )
            Nblocks_f = X_frame.shape[1]
            sum_err_f = 0.0
            coeffs_f = []

            # 对当前帧每个块执行 KSVD 编码
            for i in range(Nblocks_f):
                x = X_frame[:, i:i + 1]
                Ar = omp(Dk, x, args.sparsity)
                Ai = quantize(Ar, args.quant)
                rec = Dk @ (Ai * args.quant)
                err = float(norm(x - rec.reshape(x.shape))) ** 2
                sum_err_f += err
                coeffs_f.append(Ai.flatten().tolist())

            # 写当前帧系数
            out_dir = out_root / f"QP={args.qp}" / "KSVD" / f"frame_{frame_idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            coeff_path = out_dir / f"coeffs_{frame_idx}.txt"
            with open(coeff_path, "w") as f:
                for vec in coeffs_f:
                    f.write(" ".join(str(int(v)) for v in vec) + "\n")

            # 调用 CABAC 并统计比特
            bin_path = out_dir / f"out_{frame_idx}.bin"
            bits_frame = run_cabac(
                str(coeff_path), str(bin_path),
                w, h, args.cabac_exe, args.tu_cfg
            )

            # 累加全局
            total_bits += bits_frame
            total_err += sum_err_f

            # 记录该帧结果
            pixels_f = Nblocks_f * blk * blk
            mse_f = sum_err_f / pixels_f
            psnr_f = 10 * math.log10((255 ** 2) / mse_f) if mse_f > 0 else float('inf')
            bpp_f = bits_frame / pixels_f
            results.append([f"KSVD_f{frame_idx}", args.qp,
                            f"{psnr_f:.3f}", f"{bpp_f:.6f}"])

        # 全帧汇总指标
        total_pixels = args.test_frames * (w // blk) * (h // blk) * (blk * blk)
        mse_all = total_err / total_pixels
        psnr_all = 10 * math.log10((255 ** 2) / mse_all) if mse_all > 0 else float('inf')
        bpp_all = total_bits / total_pixels
        results.append(["KSVD_overall", args.qp,
                        f"{psnr_all:.3f}", f"{bpp_all:.6f}"])

    # ---------- MultiKSVD 多字典（按帧处理 & 记录全帧信息） ----------
    if "MultiKSVD" not in args.skip_schemes:
        print("Evaluating MultiKSVD …")
        Mk = load_dicts(args.dict_dir, "MultiKSVD")
        num_mk = len(Mk)
        idx_bits_mk = math.ceil(math.log2(num_mk))

        total_bits = 0
        total_err = 0.0

        for frame_idx in range(args.test_frames):
            # 读取并处理当前帧的残差块
            X_frame = video_to_residual_blocks(
                args.test_video, 1, blk,
                args.residual_offset, w, h,
                skip=args.skip_frames + frame_idx
            )
            Nblocks_f = X_frame.shape[1]
            sum_err_f = 0.0
            coeffs_f = []

            # 对当前帧所有块做 MultiKSVD + RD 选择
            for i in range(Nblocks_f):
                x = X_frame[:, i:i + 1]
                best = None
                best_Ai = None

                for D in Mk:
                    Ar = omp(D, x, args.sparsity)
                    Ai = quantize(Ar, args.quant)
                    rec = D @ (Ai * args.quant)
                    err = float(norm(x - rec.reshape(x.shape))) ** 2

                    bits = est.estimate_block_bits(
                        Ai.reshape((blk, blk)), update=False
                    )
                    bits_with_idx = bits + (0 if np.all(Ai == 0) else idx_bits_mk)
                    J = err + args.lam_KSVD * bits_with_idx

                    if best is None or J < best[0]:
                        best = (J, err, bits_with_idx)
                        best_Ai = Ai.copy()

                est.estimate_block_bits(best_Ai.reshape((blk, blk)), update=True)
                _, e, b_frame = best
                sum_err_f += e
                coeffs_f.append(best_Ai.flatten().tolist())

            # 写当前帧系数文件
            out_dir = out_root / f"QP={args.qp}" / "MultiKSVD" / f"frame_{frame_idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            coeff_path = out_dir / f"coeffs_{frame_idx}.txt"
            with open(coeff_path, "w") as f:
                for vec in coeffs_f:
                    f.write(" ".join(str(int(v)) for v in vec) + "\n")

            # 调用 CABAC 并统计比特
            bin_path = out_dir / f"out_{frame_idx}.bin"
            bits_frame = run_cabac(
                str(coeff_path), str(bin_path),
                w, h, args.cabac_exe, args.tu_cfg
            )
            # 累加索引比特
            nonzero_blocks = sum(1 for vec in coeffs_f if any(v != 0 for v in vec))
            bits_frame += nonzero_blocks * idx_bits_mk

            total_bits += bits_frame
            total_err += sum_err_f

            # 记录这一帧的 PSNR & bpp
            pixels_f = Nblocks_f * blk * blk
            mse_f = sum_err_f / pixels_f
            psnr_f = 10 * math.log10((255 ** 2) / mse_f) if mse_f > 0 else float('inf')
            bpp_f = bits_frame / pixels_f
            results.append([f"MultiKSVD_f{frame_idx}", args.qp,
                            f"{psnr_f:.3f}", f"{bpp_f:.6f}"])

        # 全帧汇总指标
        total_pixels = args.test_frames * (w // blk) * (h // blk) * (blk * blk)
        mse_all = total_err / total_pixels
        psnr_all = 10 * math.log10((255 ** 2) / mse_all) if mse_all > 0 else float('inf')
        bpp_all = total_bits / total_pixels
        results.append(["MultiKSVD_overall", args.qp,
                        f"{psnr_all:.3f}", f"{bpp_all:.6f}"])

    # ---------- IterMultiKSVD 多字典（按帧处理 & 记录全帧信息） ----------
    if "IterMultiKSVD" not in args.skip_schemes:
        print("Evaluating IterMultiKSVD …")
        Ik = load_dicts(args.dict_dir, "IterMultiKSVD")
        num_ik = len(Ik)
        idx_bits_ik = math.ceil(math.log2(num_ik))

        total_bits = 0
        total_err = 0.0

        for frame_idx in range(args.test_frames):
            # 只取当前帧的残差块
            X_frame = video_to_residual_blocks(
                args.test_video, 1, blk,
                args.residual_offset, w, h,
                skip=args.skip_frames + frame_idx
            )
            Nblocks_f = X_frame.shape[1]
            sum_err_f = 0.0
            coeffs_f = []

            # 对当前帧每个块执行 IterMultiKSVD + RD 选择
            for i in range(Nblocks_f):
                x = X_frame[:, i:i + 1]
                best = None
                best_Ai = None

                for D in Ik:
                    Ar = omp(D, x, args.sparsity)
                    Ai = quantize(Ar, args.quant)
                    rec = D @ (Ai * args.quant)
                    err = float(norm(x - rec.reshape(x.shape))) ** 2

                    bits = est.estimate_block_bits(
                        Ai.reshape((blk, blk)), update=False
                    )
                    bits_with_idx = bits + (0 if np.all(Ai == 0) else idx_bits_ik)
                    J = err + args.lam_KSVD * bits_with_idx

                    if best is None or J < best[0]:
                        best = (J, err, bits_with_idx)
                        best_Ai = Ai.copy()

                est.estimate_block_bits(best_Ai.reshape((blk, blk)), update=True)
                _, e, b_frame = best
                sum_err_f += e
                coeffs_f.append(best_Ai.flatten().tolist())

            # 写当前帧系数文件
            out_dir = out_root / f"QP={args.qp}" / "IterMultiKSVD" / f"frame_{frame_idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            coeff_path = out_dir / f"coeffs_{frame_idx}.txt"
            with open(coeff_path, "w") as f:
                for vec in coeffs_f:
                    f.write(" ".join(str(int(v)) for v in vec) + "\n")

            # 调用 CABAC 并统计比特
            bin_path = out_dir / f"out_{frame_idx}.bin"
            bits_frame = run_cabac(
                str(coeff_path), str(bin_path),
                w, h, args.cabac_exe, args.tu_cfg
            )
            # 累加索引比特
            nonzero_blocks = sum(1 for vec in coeffs_f if any(v != 0 for v in vec))
            bits_frame += nonzero_blocks * idx_bits_ik

            total_bits += bits_frame
            total_err += sum_err_f

            # 记录该帧 PSNR & bpp
            pixels_f = Nblocks_f * blk * blk
            mse_f = sum_err_f / pixels_f
            psnr_f = 10 * math.log10((255 ** 2) / mse_f) if mse_f > 0 else float('inf')
            bpp_f = bits_frame / pixels_f
            results.append([f"IterMultiKSVD_f{frame_idx}", args.qp,
                            f"{psnr_f:.3f}", f"{bpp_f:.6f}"])

        # 全帧汇总指标
        total_pixels = args.test_frames * (w // blk) * (h // blk) * (blk * blk)
        mse_all = total_err / total_pixels
        psnr_all = 10 * math.log10((255 ** 2) / mse_all) if mse_all > 0 else float('inf')
        bpp_all = total_bits / total_pixels
        results.append(["IterMultiKSVD_overall", args.qp,
                        f"{psnr_all:.3f}", f"{bpp_all:.6f}"])

    # ---------- 写入 summary.csv ----------
    csv_path = out_root / "summary.csv"
    header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["Scheme", "QP", "PSNR(dB)", "bpp"])
        for row in results:
            w.writerow(row)

    print("Done. Results →", csv_path)

if __name__=="__main__":
    main()
