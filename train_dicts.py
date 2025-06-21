#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本：KSVD / Multi-KSVD / Iter-Multi-KSVD 字典学习并保存
"""

import os
import argparse
import pathlib
import warnings
import math

import joblib
import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import norm, svd
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.cluster import KMeans
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
# ------------------------------------------------------------
# 简化版 HEVC CABAC 比特数估计器
# ------------------------------------------------------------
class HevcCabacEstimator:
    def __init__(self):
        self.init_scan_tables()

    # ---------------- 内部辅助 ----------------
    def init_scan_tables(self):
        self.diag_scan_4x4 = [
            0, 1, 4, 8, 5, 2, 3, 6,
            9, 12, 13, 10, 7, 11, 14, 15
        ]
        self.diag_scan_8x8 = self._generate_diag_scan(8)

    def _generate_diag_scan(self, size: int):
        scan_order = []
        for sum_val in range(2 * size - 1):
            if sum_val < size:
                y = sum_val
                x = 0
            else:
                y = size - 1
                x = sum_val - size + 1
            while x < size and y >= 0:
                if x < size and y < size:
                    scan_order.append(y * size + x)
                x += 1
                y -= 1
        return scan_order

    # ---------------- 估计子函数 ----------------
    @staticmethod
    def estimate_entropy_bits(prob_lps: float) -> float:
        if prob_lps <= 0.0 or prob_lps >= 1.0:
            return 0.0
        prob_mps = 1.0 - prob_lps
        return -prob_lps * math.log2(prob_lps) - prob_mps * math.log2(prob_mps)

    def estimate_sig_coeff_flag(self, coeff: int, pos_x: int, pos_y: int,
                                log2_blk_size: int) -> float:
        if coeff == 0:
            return 0.0
        return self.estimate_entropy_bits(0.5)   # 简化：固定 0.5

    def estimate_last_sig_pos(self, last_x: int, last_y: int,
                              log2_blk_size: int) -> float:
        bits = 0.0
        # x 方向
        if last_x > 3:
            bits += 3.0
            bits += math.ceil(math.log2(last_x - 3 + 1))
        else:
            bits += last_x * 0.5
        # y 方向
        if last_y > 3:
            bits += 3.0
            bits += math.ceil(math.log2(last_y - 3 + 1))
        else:
            bits += last_y * 0.5
        return bits

    def estimate_coeff_level(self, abs_level: int, pos_in_cg: int,
                             first_c2_flag_idx: int) -> float:
        bits = 0.0
        if abs_level > 0:
            bits += 0.8            # coeff_abs_sign_flag
            if abs_level > 1 and pos_in_cg == first_c2_flag_idx:
                bits += 0.7        # coeff_abs_level_greater1_flag
                if abs_level > 2:
                    remain = abs_level - 3
                    if remain < 4:
                        bits += 2.0
                    else:
                        bits += 2.0 + math.ceil(math.log2(remain - 3))
            elif abs_level == 2 and pos_in_cg == first_c2_flag_idx:
                bits += 0.7
            bits += 1.0             # sign bit
        return bits

    # ---------------- 主函数：整块估计 ----------------
    def estimate_residual_block(self, coeffs: np.ndarray,
                                log2_blk_size: int) -> float:
        blk_size = 1 << log2_blk_size
        total_bits = 0.0
        if np.all(coeffs == 0):
            return 1.0              # 仅 coded_block_flag
        total_bits += 1.0           # coded_block_flag

        # 选扫描顺序
        if blk_size == 4:
            scan_order = self.diag_scan_4x4
        elif blk_size == 8:
            scan_order = self.diag_scan_8x8
        else:
            scan_order = self._generate_diag_scan(blk_size)

        coeffs_1d = coeffs.flatten()[scan_order]
        # 找最后一个非零
        last_nz_pos = -1
        for i in range(len(coeffs_1d) - 1, -1, -1):
            if coeffs_1d[i] != 0:
                last_nz_pos = i
                break
        if last_nz_pos < 0:
            return total_bits

        last_flat_idx = scan_order[last_nz_pos]
        last_y = last_flat_idx // blk_size
        last_x = last_flat_idx % blk_size
        total_bits += self.estimate_last_sig_pos(last_x, last_y, log2_blk_size)

        cg_size = 4
        first_c2_flag_idx = -1
        for scan_pos in range(last_nz_pos, -1, -1):
            flat_idx = scan_order[scan_pos]
            pos_y = flat_idx // blk_size
            pos_x = flat_idx % blk_size
            abs_level = abs(int(coeffs_1d[scan_pos]))

            if scan_pos < last_nz_pos:
                total_bits += self.estimate_sig_coeff_flag(
                    abs_level, pos_x, pos_y, log2_blk_size
                )

            if abs_level > 0:
                pos_in_cg = (pos_y % cg_size) * cg_size + (pos_x % cg_size)
                if first_c2_flag_idx < 0:
                    first_c2_flag_idx = pos_in_cg
                total_bits += self.estimate_coeff_level(
                    abs_level, pos_in_cg, first_c2_flag_idx
                )

        return total_bits

# -----------------------------------------------------------------------------
# —— 辅助函数：（与原脚本保持一致，略去已知易错句）
#    video_to_residual_blocks, omp, quantize,
#    init_dict, enforce_norm, _update_atom, k_svd,
#    iter_rd_multicsvd_train
# ------------------------------------------------------------
# DCT / DST-7 / DCT-8 字典（备用）
# ------------------------------------------------------------
def dct_2d_dict(blk):
    N = blk
    D = np.zeros((N*N, N*N))
    idx = 0
    for u in range(N):
        for v in range(N):
            cu = 1/np.sqrt(2) if u == 0 else 1.0
            cv = 1/np.sqrt(2) if v == 0 else 1.0
            basis = np.outer(
                np.cos((np.pi*(2*np.arange(N)+1)*u)/(2*N)),
                np.cos((np.pi*(2*np.arange(N)+1)*v)/(2*N))
            )
            D[:, idx] = ((2/N)*cu*cv * basis).flatten()
            idx += 1
    return D / (norm(D, axis=0, keepdims=True) + 1e-12)

def dst7_2d_dict(blk):
    N = blk
    D = np.zeros((N*N, N*N))
    idx = 0
    for u in range(N):
        for v in range(N):
            basis = np.outer(
                np.sin((np.pi*(2*np.arange(N)+1)*(u+1))/(2*N+1)),
                np.sin((np.pi*(2*np.arange(N)+1)*(v+1))/(2*N+1))
            )
            D[:, idx] = (np.sqrt(2/(2*N+1)) * basis).flatten()
            idx += 1
    return D / (norm(D, axis=0, keepdims=True) + 1e-12)

def dct8_2d_dict(blk):
    N = blk
    D = np.zeros((N*N, N*N))
    idx = 0
    for u in range(N):
        for v in range(N):
            basis = np.outer(
                np.cos((np.pi*(2*np.arange(N)+1)*(2*u+1))/(4*N)),
                np.cos((np.pi*(2*np.arange(N)+1)*(2*v+1))/(4*N))
            )
            D[:, idx] = (np.sqrt(2/N) * basis).flatten()
            idx += 1
    return D / (norm(D, axis=0, keepdims=True) + 1e-12)
# ------------------------------------------------------------
# 残差块提取
# ------------------------------------------------------------
def video_to_residual_blocks(path, num_frames, blk, offset, width, height, skip=1):
    """
    从 4:2:0 YUV420 文件提取 Y 残差块，返回 shape=(blk*blk, N) 的矩阵
    """
    y_size   = width * height
    uv_size  = y_size // 4
    frame_sz = y_size + 2 * uv_size

    blocks = []
    with open(path, "rb") as f:
        f.seek(frame_sz * skip, os.SEEK_CUR)           # 跳首帧
        for _ in range(num_frames):
            data = f.read(frame_sz)
            if len(data) < frame_sz:
                break
            Y = (np.frombuffer(data, count=y_size, dtype=np.uint8)
                   .reshape((height, width)).astype(np.float64) - offset)
            bh, bw = (height // blk) * blk, (width // blk) * blk
            B = (Y[:bh, :bw]
                 .reshape(bh//blk, blk, bw//blk, blk)
                 .swapaxes(1, 2)
                 .reshape(-1, blk * blk).T)
            blocks.append(B)
    if not blocks:
        raise RuntimeError(f"No valid frames in {path}")
    return np.hstack(blocks)

# ------------------------------------------------------------
# OMP / 量化
# ------------------------------------------------------------
def omp(D, X, k):
    G  = D.T @ D
    DX = D.T @ X
    return orthogonal_mp_gram(G, DX, n_nonzero_coefs=k).astype(np.float64)

def quantize(A_real, q):
    return np.round(A_real / q).astype(int)
# ------------------------------------------------------------
# K-SVD 辅助
# ------------------------------------------------------------
def init_dict(X, K, scale=1.0, init='dct'):
    """
    返回初始化字典 (m, K)
    """
    m   = X.shape[0]
    blk = int(np.sqrt(m))
    if init == 'dct':
        D0 = dct_2d_dict(blk)[:, :K] if K <= m else dct_2d_dict(blk)
        if K > m:
            D0 = np.hstack([D0, np.random.randn(m, K-m)])
    elif init == 'data':
        cols = np.random.choice(X.shape[1], K, replace=False)
        D0 = X[:, cols].copy()
    else:  # random
        D0 = np.random.randn(m, K)
    D0 /= (norm(D0, axis=0, keepdims=True) + 1e-12)
    return D0 * scale

def enforce_norm(D, target):
    D /= (norm(D, axis=0, keepdims=True) / target + 1e-12)

def _update_atom(args):
    j, D, A, X, target, dead, seed = args
    rng = np.random.default_rng(seed)
    idxs = np.nonzero(A[j])[0]
    if idxs.size == 0:                    # “死”原子
        dead[j] += 1
        if dead[j] >= 3:
            D[:, j] = X[:, rng.integers(0, X.shape[1])] * target
            dead[j] = 0
        return D[:, j], None
    dead[j] = 0
    E = X[:, idxs] - D @ A[:, idxs] + np.outer(D[:, j], A[j, idxs])
    U, s, Vt = svd(E, full_matrices=False)
    d_new = U[:, 0] * target
    coef  = s[0] * Vt[0]
    coef_q = np.round(coef / target).astype(int) if target != 1 else coef
    return d_new, (idxs, coef_q)

def k_svd(X, K, sparsity, iters, n_jobs, init='dct'):
    """
    经典 K-SVD：返回 (m, K) 字典
    """
    D = init_dict(X, K, 1.0, init)
    dead = np.zeros(K, dtype=int)
    for _ in tqdm(range(iters), desc="K-SVD", leave=False):
        A = omp(D, X, sparsity)
        args_list = [(j, D, A, X, 1.0, dead, 1000 + j) for j in range(K)]
        updates = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_update_atom)(arg) for arg in args_list
        )
        for j, (d_new, info) in enumerate(updates):
            D[:, j] = d_new
            if info:
                idxs, coef_q = info
                A[j, idxs] = coef_q
        enforce_norm(D, 1.0)
    return D


# ======================================================================
#  新增：Iter-RD-Multi-KSVD 训练（外循环）
# ======================================================================
def iter_rd_multicsvd_train(X_train,
                            blk,
                            sparsity,
                            num_dicts,
                            outer_iter,
                            ksvd_inner_iter,
                            n_jobs,
                            quant,
                            lam):
    """
    训练 num_dicts 个字典；每轮结束后按照 RD 成本重新划分训练块，
    并移除那些最优结果为全零的块。
    首轮对空簇使用 DCT 基而非随机初始化。
    """
    # ---------- 辅助：增量 K-SVD ----------
    def k_svd_warm(X, D_init, sparsity, iters, n_jobs):
        D   = D_init.copy()
        K   = D.shape[1]
        dead = np.zeros(K, dtype=int)
        for _ in range(iters):
            A = omp(D, X, sparsity)
            args_list = [(j, D, A, X, 1.0, dead, 2000 + j)
                         for j in range(K)]
            updates = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_update_atom)(arg) for arg in args_list
            )
            for j, (d_new, info) in enumerate(updates):
                D[:, j] = d_new
                if info:
                    idxs, coef_q = info
                    A[j, idxs] = coef_q
            enforce_norm(D, 1.0)
        return D

    # ---------- 预备变量 ----------
    m         = blk * blk
    log2_blk  = int(math.log2(blk))
    est       = HevcCabacEstimator()
    dict_bits = math.ceil(math.log2(num_dicts))
    rng       = np.random.default_rng(0)

    # 生成一次 DCT 基，shape=(m, m)
    Dct_base = dct_2d_dict(blk)

    # 0) 初始聚类得到标签
    km     = KMeans(n_clusters=num_dicts, random_state=0).fit(X_train.T)
    labels = km.labels_

    # 持久化字典列表
    D_list = [None] * num_dicts

    # 外循环
    for outer in range(outer_iter):
        # 1) 训练 / 继续训练各簇字典
        for c in range(num_dicts):
            Xi = X_train[:, labels == c]
            # 空簇
            if Xi.shape[1] == 0:
                if outer == 0:
                    # 第1轮用 DCT 基
                    Dc = Dct_base.copy()
                else:
                    # 后续轮次仍用随机
                    Dc = rng.standard_normal((m, m))
                    Dc /= (norm(Dc, axis=0, keepdims=True) + 1e-12)
            else:
                # 第1轮如果还没字典，初始化为 DCT 基
                if D_list[c] is None and outer == 0:
                    Dc = Dct_base.copy()
                elif D_list[c] is None:
                    # 首轮后非空簇第1次 K-SVD
                    Dc = k_svd(Xi, m, sparsity, ksvd_inner_iter, n_jobs)
                else:
                    # 后续轮次增量 K-SVD
                    Dc = k_svd_warm(Xi, D_list[c], sparsity,
                                     ksvd_inner_iter, n_jobs)
            D_list[c] = Dc

        # 2) RD 重新分配训练块 & 移除全零块
        new_labels = []
        keep_indices = []
        for i in range(X_train.shape[1]):
            x = X_train[:, i:i+1]
            best_J, best_idx = float('inf'), -1
            best_Ai = None

            for idx, D in enumerate(D_list):
                A_r = omp(D, x, sparsity)
                A_i = quantize(A_r, quant)
                rec = D @ (A_i * quant)
                err = float(norm(x - rec.reshape(x.shape)))**2

                bits_est = est.estimate_residual_block(
                    A_i.reshape((blk, blk)), log2_blk_size=log2_blk
                )
                if np.any(A_i != 0):
                    bits_est += dict_bits

                J = err + lam * bits_est
                if J < best_J:
                    best_J, best_idx = J, idx
                    best_Ai = A_i

            # 只保留非全零块
            if np.any(best_Ai != 0):
                new_labels.append(best_idx)
                keep_indices.append(i)

        # 没有块被保留时退出
        if not keep_indices:
            print(f"[Iter-Multi-KSVD] All blocks zero at iter {outer+1}.")
            break

        # 更新训练集与标签
        X_train = X_train[:, keep_indices]
        labels = np.array(new_labels, dtype=int)

        # # 收敛判据：标签未变化
        # if outer > 0 and np.array_equal(labels, new_labels):
        #     print(f"[Iter-Multi-KSVD] Converged at outer-iter "
        #           f"{outer+1}/{outer_iter}")
        #     break

    return D_list


# -----------------------------------------------------------------------------
# （此处省略，直接拷贝原脚本中同名函数）

def save_dicts(output_dir, method_name, dict_list):
    """
    将字典或字典列表保存为 .npy 文件
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(dict_list, list):
        for idx, D in enumerate(dict_list):
            np.save(output_dir / f"{method_name}_dict_{idx}.npy", D)
    else:
        # 单个字典
        np.save(output_dir / f"{method_name}.npy", dict_list)

def main():
    ap = argparse.ArgumentParser(
        description="训练并保存 KSVD / Multi-KSVD / Iter-Multi-KSVD 字典"
    )
    ap.add_argument("--train_video", nargs="+", required=True, help="训练用视频序列（可多个）")
    ap.add_argument("--train_frames", type=int, default=1)
    ap.add_argument("--skip_frames", type=int, default=1)
    ap.add_argument("--widths",  nargs="+", type=int, required=True, help="每个视频的宽度（顺序对应）")
    ap.add_argument("--heights", nargs="+", type=int, required=True, help="每个视频的高度（顺序对应）")
    ap.add_argument("--residual_offset", type=int, default=128)
    ap.add_argument("--block",   type=int, default=8)
    ap.add_argument("--sparsity",type=int, default=3)
    ap.add_argument("--iter",       type=int, default=20)
    ap.add_argument("--inter_iter", type=int, default=1)
    ap.add_argument("--outer_iter", type=int, default=20)
    ap.add_argument("--num_dicts",  type=int, default=4)
    ap.add_argument("--train_blocks", type=int, default=None,
                    help="随机抽取块数量（可选）")
    ap.add_argument("--n_jobs", type=int, default=os.cpu_count())
    ap.add_argument("--output_dir", default="trained_dicts")
    ap.add_argument("--lam", type=float, default=100.0,help="外部指定的 λ 权重（实数）")
    ap.add_argument("--quant", type=int, default=1)
    args = ap.parse_args()

    # 校验参数长度一致
    if not (len(args.train_video) == len(args.widths) == len(args.heights)):
        raise ValueError("train_video、widths、heights数量必须一致！")

    # 1) 提取训练残差块
    print("Extracting training residual blocks …")
    Xtr_list = []
    for vid, w, h in zip(args.train_video, args.widths, args.heights):
        Xtr_list.append(
            video_to_residual_blocks(
                vid, args.train_frames, args.block,
                args.residual_offset, w, h, args.skip_frames
            )
        )
    X_train = np.hstack(Xtr_list)


    # 2) 过滤与子采样
    thr = 0.5 * args.quant
    mask = np.linalg.norm(X_train, axis=0) >= thr
    X_train = X_train[:, mask]
    if args.train_blocks:
        perm = np.random.permutation(X_train.shape[1])[: args.train_blocks]
        X_train = X_train[:, perm]
    print(f"Final training blocks: {X_train.shape}")

    m = args.block * args.block

    # ———— KSVD ————
    print("Training single-dict KSVD …")
    Dk = k_svd(X_train, m, args.sparsity, args.iter, args.n_jobs)
    save_dicts(args.output_dir, "KSVD", Dk)

    # ———— Multi-KSVD ————
    print(f"Training Multi-KSVD ({args.num_dicts} dicts) …")
    km = KMeans(n_clusters=args.num_dicts, random_state=0).fit(X_train.T)
    D_list = []
    for c in range(args.num_dicts):
        Xi = X_train[:, km.labels_ == c]
        Dc = k_svd(
            Xi if Xi.shape[1] else X_train[:, :m],
            m, args.sparsity, args.iter, args.n_jobs
        )
        D_list.append(Dc)
    save_dicts(args.output_dir, f"MultiKSVD_{args.num_dicts}_lam{args.lam}", D_list)

    # ———— Iter-Multi-KSVD ————
    print(f"Training Iter-Multi-KSVD ({args.num_dicts} dicts, {args.outer_iter} outer-iters) …")
    D_iter = iter_rd_multicsvd_train(
        X_train, args.block, args.sparsity,
        args.num_dicts, args.outer_iter, args.inter_iter,
        args.n_jobs, quant=args.quant, lam=args.lam
    )
    save_dicts(args.output_dir, f"IterMultiKSVD_{args.num_dicts}", D_iter)

    print("All dictionaries saved to:", args.output_dir)

if __name__ == "__main__":
    main()
