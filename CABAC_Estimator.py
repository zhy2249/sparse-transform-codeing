from __future__ import annotations
from typing import List, Sequence
from enum import Enum, auto
import math
import copy

# Precomputed CABAC tables from HM's ContextModel.cpp (FAST_BIT_EST path)
# Each entry corresponds to 15-bit fixed point fractional bits.
NEXT_STATE_MPS = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
    98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
    114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 124, 125, 126, 127
]

NEXT_STATE_LPS = [
    1, 0, 0, 1, 2, 3, 4, 5, 4, 5, 8, 9, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 18, 19, 22, 23, 22, 23, 24, 25,
    26, 27, 26, 27, 30, 31, 30, 31, 32, 33, 32, 33, 36, 37, 36, 37,
    38, 39, 38, 39, 42, 43, 42, 43, 44, 45, 44, 45, 46, 47, 48, 49,
    48, 49, 50, 51, 52, 53, 52, 53, 54, 55, 54, 55, 56, 57, 58, 59,
    58, 59, 60, 61, 60, 61, 60, 61, 62, 63, 64, 65, 64, 65, 66, 67,
    66, 67, 66, 67, 68, 69, 68, 69, 70, 71, 70, 71, 70, 71, 72, 73,
    72, 73, 72, 73, 74, 75, 74, 75, 74, 75, 76, 77, 76, 77, 126, 127
]

# Entropy bits table (FAST_BIT_EST) scaled by 2^15.
ENTROPY_BITS = [
    0x07b23, 0x085f9, 0x074a0, 0x08cbc, 0x06ee4, 0x09354, 0x067f4, 0x09c1b,
    0x060b0, 0x0a62a, 0x05a9c, 0x0af5b, 0x0548d, 0x0b955, 0x04f56, 0x0c2a9,
    0x04a87, 0x0cbf7, 0x045d6, 0x0d5c3, 0x04144, 0x0e01b, 0x03d88, 0x0e937,
    0x039e0, 0x0f2cd, 0x03663, 0x0fc9e, 0x03347, 0x10600, 0x03050, 0x10f95,
    0x02d4d, 0x11a02, 0x02ad3, 0x12333, 0x0286e, 0x12cad, 0x02604, 0x136df,
    0x02425, 0x13f48, 0x021f4, 0x149c4, 0x0203e, 0x1527b, 0x01e4d, 0x15d00,
    0x01c99, 0x166de, 0x01b18, 0x17017, 0x019a5, 0x17988, 0x01841, 0x18327,
    0x016df, 0x18d50, 0x015d9, 0x19547, 0x0147c, 0x1a083, 0x0138e, 0x1a8a3,
    0x01251, 0x1b418, 0x01166, 0x1bd27, 0x01068, 0x1c77b, 0x00f7f, 0x1d18e,
    0x00eda, 0x1d91a, 0x00e19, 0x1e254, 0x00d4f, 0x1ec9a, 0x00c90, 0x1f6e0,
    0x00c01, 0x1fef8, 0x00b5f, 0x208b1, 0x00ab6, 0x21362, 0x00a15, 0x21e46,
    0x00988, 0x2285d, 0x00934, 0x22ea8, 0x008a8, 0x239b2, 0x0081d, 0x24577,
    0x007c9, 0x24ce6, 0x00763, 0x25663, 0x00710, 0x25e8f, 0x006a0, 0x26a26,
    0x00672, 0x26f23, 0x005e8, 0x27ef8, 0x005ba, 0x284b5, 0x0055e, 0x29057,
    0x0050c, 0x29bab, 0x004c1, 0x2a674, 0x004a7, 0x2aa5e, 0x0046f, 0x2b32f,
    0x0041f, 0x2c0ad, 0x003e7, 0x2ca8d, 0x003ba, 0x2d323, 0x0010c, 0x3bfbb
]

# Context initialisation tables extracted from HM for P-slices (luma only)
# They roughly correspond to the initial CABAC states used for residual coding.
SIG_FLAG_INIT = [
    155, 154, 139, 153, 139, 123, 123, 63,
    153, 166, 183, 140, 136, 153, 154,
    166, 183, 140, 136, 153, 154, 166,
    183, 140, 136, 153, 154, 140,
]

ONE_FLAG_INIT = [
    154, 196, 196, 167, 154, 152, 167, 182,
    182, 134, 149, 136, 153, 121, 136, 137,
]

ABS_FLAG_INIT = [107, 167, 91, 122]

# Significance coefficient group flag contexts (first, others)
SIG_CG_FLAG_INIT = [121, 140]

# Last-significant-position context initialisation (P-slice, luma)
LAST_POS_INIT = [
    125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94,
]

# Group index tables for last position coding
G_UI_GROUP_IDX = [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
                  8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9]
G_UI_MIN_IN_GROUP = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24]

# Significance map context mapping for 4x4 blocks
CTX_IND_MAP_4x4 = [
    0, 1, 4, 5,
    2, 3, 4, 5,
    6, 6, 8, 8,
    7, 7, 8, 8,
]

# Context set starting offsets for significance map (luma, 4x4 and 8x8)
SIG_CTX_SET_START = [0, 9, 21, 27]
NOT_FIRST_GROUP_OFFSET = 3


def init_state(qp: int, init_value: int) -> int:
    """Convert HM init value to CABAC state."""
    slope = ((init_value >> 4) * 5) - 45
    offset = ((init_value & 15) << 3) - 16
    init = min(max(1, ((slope * qp) >> 4) + offset), 126)
    mps = 1 if init >= 64 else 0
    state = ((init - 64) if mps else (63 - init)) << 1
    return state + mps


def init_contexts(values: List[int], qp: int = 22) -> List[CabacContext]:
    return [CabacContext(init_state(qp, v)) for v in values]


def convert_to_bit(x: int) -> int:
    """Return log2(x) - 2 for block sizes (4,8,...)."""
    c = -1
    i = 4
    while i <= x:
        c += 1
        i <<= 1
    return max(c, 0)


def last_ctx_params(width: int, height: int) -> tuple[int, int, int, int]:
    """Derive offsets and shifts for last-position contexts."""
    cw = convert_to_bit(width)
    ch = convert_to_bit(height)
    off_x = (cw * 3) + ((cw + 1) >> 2)
    off_y = (ch * 3) + ((ch + 1) >> 2)
    shift_x = (cw + 3) >> 2
    shift_y = (ch + 3) >> 2
    return off_x, off_y, shift_x, shift_y


def sig_ctx_idx(x: int, y: int, width: int, height: int) -> int:
    """Return context index for significance flag at position (x,y)."""
    if width == 4 and height == 4:
        return CTX_IND_MAP_4x4[y * 4 + x]
    pos_x_in = x & 3
    pos_y_in = y & 3
    pos_sum = pos_x_in + pos_y_in
    if pos_sum >= 3:
        cnt = 0
    elif pos_sum >= 1:
        cnt = 1
    else:
        cnt = 2
    not_first = ((x >> 2) + (y >> 2)) > 0
    return SIG_CTX_SET_START[1] + (NOT_FIRST_GROUP_OFFSET if not_first else 0) + cnt

class CabacContext:
    """Simplified CABAC context model for bit estimation."""
    def __init__(self, init_state: int = 0):
        self.state = init_state

    def bits(self, bin_value: int) -> float:
        return ENTROPY_BITS[self.state ^ bin_value] / float(1 << 15)

    def update(self, bin_value: int):
        """Update context state after coding a bin."""
        mps = self.state & 1
        self.state = NEXT_STATE_MPS[self.state] if bin_value == mps else NEXT_STATE_LPS[self.state]


COEF_REMAIN_BIN_REDUCTION = 3
C1FLAG_NUMBER = 8
C2FLAG_NUMBER = 1


class ResetMode(Enum):
    """Granularity for context resets."""
    CTU = auto()
    TU = auto()
    FRAME = auto()


def golomb_rice_bits(symbol: int, k: int = 0) -> float:
    """Return bit count for ``symbol`` coded with Golomb-Rice."""
    if symbol < (COEF_REMAIN_BIN_REDUCTION << k):
        prefix = symbol >> k
        return prefix + 1 + k
    symbol -= COEF_REMAIN_BIN_REDUCTION << k
    length = k
    while symbol >= (1 << length):
        symbol -= 1 << length
        length += 1
    return COEF_REMAIN_BIN_REDUCTION + length + 1 - k + length


def update_gr_k(k: int, symbol: int) -> int:
    """Update Golomb-Rice parameter ``k`` following HM's rule."""
    if symbol > (3 << k):
        k = min(k + 1, 4)
    elif k > 0 and symbol < (1 << (k - 1)):
        k -= 1
    return k


def bypass_bits(num: int = 1) -> float:
    """Return bit count for ``num`` bypass-coded bits."""
    return float(num)


def sign_bit_bits(hidden: bool = False) -> float:
    """Return bit count for coding a sign (bypass path)."""
    return 0.0 if hidden else 1.0


def level_bits(level: int, sign_hidden: bool, ctx_set: dict,
               one_ctx: CabacContext, abs_ctx: CabacContext,
               k: int, c1_idx: int, c2_idx: int) -> float:
    """Estimate bits for coding ``level`` with sign."""
    bits = sign_bit_bits(sign_hidden)

    base_level = 1
    if c1_idx < C1FLAG_NUMBER:
        base_level = 2 + (c2_idx < C2FLAG_NUMBER)

    if level >= base_level:
        symbol = level - base_level
        if c1_idx < C1FLAG_NUMBER:
            bits += one_ctx.bits(1)
            one_ctx.update(1)
            if c2_idx < C2FLAG_NUMBER:
                bits += abs_ctx.bits(1)
                abs_ctx.update(1)
        bits += golomb_rice_bits(symbol, k)
    elif level == 1:
        bits += one_ctx.bits(0)
        one_ctx.update(0)
    elif level == 2:
        bits += one_ctx.bits(1)
        one_ctx.update(1)
        bits += abs_ctx.bits(0)
        abs_ctx.update(0)
    return bits


def diag_scan_order(width: int, height: int) -> List[int]:
    """Generate diagonal scan order."""
    order = []
    x = y = 0
    for _ in range(width * height):
        order.append(y * width + x)
        if x == width - 1 or y == 0:
            y += x + 1
            x = 0
            if y >= height:
                x += y - (height - 1)
                y = height - 1
        else:
            x += 1
            y -= 1
    return order


def scan_order(size: int) -> List[int]:
    """Return scanning order for ``size`` square block."""
    if size == 4:
        return diag_scan_order(4, 4)
    # size == 8
    group_order = diag_scan_order(2, 2)
    inside = diag_scan_order(4, 4)
    order: List[int] = []
    for g in group_order:
        gx, gy = g % 2, g // 2
        base = gy * 4 * size + gx * 4
        for p in inside:
            px, py = p % 4, p // 4
            order.append(base + py * size + px)
    return order


class CabacEstimator:
    """Bit estimator that keeps contexts between blocks."""

    def __init__(self, qp: int = 22, reset_mode: ResetMode = ResetMode.CTU):
        self.qp = qp
        self.reset_mode = reset_mode
        self._init_tables()

    def _init_set(self, size: int) -> dict:
        """Initialise a context set for a transform size."""
        num_groups = (size * size + 15) // 16
        cg_ctxs = [
            CabacContext(init_state(self.qp, SIG_CG_FLAG_INIT[0] if i == 0 else SIG_CG_FLAG_INIT[1]))
            for i in range(num_groups)
        ]
        return {
            "sig": init_contexts(SIG_FLAG_INIT, self.qp),
            "one": init_contexts(ONE_FLAG_INIT, self.qp),
            "abs": init_contexts(ABS_FLAG_INIT, self.qp),
            "cg": cg_ctxs,
            "last_x": init_contexts(LAST_POS_INIT, self.qp),
            "last_y": init_contexts(LAST_POS_INIT, self.qp),
            "k": 0,
        }

    def _init_tables(self):
        self.ctx_sets = {sz: self._init_set(sz) for sz in (4, 8)}

    def _last_pos_bits(self, ctx_set: dict, x: int, y: int, w: int, h: int) -> float:
        """Estimate bits for last significant coefficient position."""
        off_x, off_y, sh_x, sh_y = last_ctx_params(w, h)

        bits = 0.0

        g_idx_x = G_UI_GROUP_IDX[x]
        g_idx_y = G_UI_GROUP_IDX[y]

        max_x = G_UI_GROUP_IDX[w - 1]
        max_y = G_UI_GROUP_IDX[h - 1]

        for ctx in range(g_idx_x):
            c = ctx_set["last_x"][off_x + (ctx >> sh_x)]
            bits += c.bits(1)
            c.update(1)
        if g_idx_x < max_x:
            c = ctx_set["last_x"][off_x + (g_idx_x >> sh_x)]
            bits += c.bits(0)
            c.update(0)

        for ctx in range(g_idx_y):
            c = ctx_set["last_y"][off_y + (ctx >> sh_y)]
            bits += c.bits(1)
            c.update(1)
        if g_idx_y < max_y:
            c = ctx_set["last_y"][off_y + (g_idx_y >> sh_y)]
            bits += c.bits(0)
            c.update(0)
        if g_idx_x > 3:
            suffix_len = (g_idx_x - 2) >> 1
            bits += bypass_bits(suffix_len)
        if g_idx_y > 3:
            suffix_len = (g_idx_y - 2) >> 1
            bits += bypass_bits(suffix_len)

        return bits

    def reset(self):
        """Reset all contexts to their initial states."""
        self._init_tables()

    def start_frame(self):
        if self.reset_mode == ResetMode.FRAME:
            self.reset()

    def start_ctu(self):
        if self.reset_mode in (ResetMode.CTU, ResetMode.TU):
            self.reset()

    def start_tu(self):
        if self.reset_mode == ResetMode.TU:
            self.reset()

    def finish_frame_bits(self) -> float:
        """Return termination and alignment overhead for a frame."""
        return 8.0

    def estimate_block_bits(self, coeffs: Sequence[Sequence[int]], update: bool = True) -> float:
        """Estimate bits for a quantized block.

        Parameters
        ----------
        coeffs : Sequence[Sequence[int]]
            量化后的系数矩阵。
        update : bool, optional
            若为 ``True`` (默认) 则在估计过程中更新内部上下文；
            否则仅计算比特数而不改变上下文状态。
        """
        if not update:
            backup = copy.deepcopy(self.ctx_sets)
        self.start_tu()
        h = len(coeffs)
        w = len(coeffs[0])
        assert h == w and h in (4, 8)

        scan = scan_order(w)
        flat = [coeffs[p // w][p % w] for p in scan]
        last_idx = -1
        for i, v in enumerate(flat):
            if v != 0:
                last_idx = i
        if last_idx < 0:
            return 0.0

        last_x = scan[last_idx] % w
        last_y = scan[last_idx] // w

        ctx_set = self.ctx_sets[w]
        total_bits = self._last_pos_bits(ctx_set, last_x, last_y, w, h)
        group_size = 16
        c1 = 1
        c2 = 0
        c1_idx = 0
        c2_idx = 0
        k = ctx_set["k"]
        ctx_set_idx = 0

        for g_start in range(0, len(scan), group_size):
            g_end = min(g_start + group_size, len(scan))
            group_idx = g_start // group_size
            group_has = any(flat[i] != 0 for i in range(g_start, g_end))
            nz_indices = [i for i in range(g_start, g_end) if flat[i] != 0]
            sign_free_idx = nz_indices[-1] if nz_indices else -1

            cg_ctx = ctx_set["cg"][group_idx]
            total_bits += cg_ctx.bits(int(group_has))
            cg_ctx.update(int(group_has))

            ctx_set_idx = (0 if group_idx == 0 else 2) + (1 if c1 == 0 else 0)
            if not group_has:
                if g_end - 1 >= last_idx:
                    break
                c1 = 1
                c2 = 0
                c1_idx = 0
                c2_idx = 0
                continue

            for idx in range(g_start, g_end):
                x = scan[idx] % w
                y = scan[idx] // w
                coef = flat[idx]
                bin_val = 1 if coef != 0 else 0
                ctx = ctx_set["sig"][sig_ctx_idx(x, y, w, h)]
                total_bits += ctx.bits(bin_val)
                ctx.update(bin_val)

                if bin_val:
                    level = abs(coef)
                    one_idx = (ctx_set_idx * 4) + min(c1_idx, 3)
                    abs_idx = ctx_set_idx
                    one_ctx = ctx_set["one"][one_idx]
                    abs_ctx = ctx_set["abs"][abs_idx]

                    k_enc = k
                    if level > (3 << k_enc):
                        k_enc += 1

                    hidden = idx == sign_free_idx
                    total_bits += level_bits(level, hidden, ctx_set,
                                            one_ctx, abs_ctx, k_enc,
                                            c1_idx, c2_idx)
                    base_lvl = 1
                    if c1_idx < C1FLAG_NUMBER:
                        base_lvl = 2 + (c2_idx < C2FLAG_NUMBER)
                    escape = max(level - base_lvl, 0)
                    k = update_gr_k(k, escape)

                    c1_idx += 1
                    if level > 1:
                        c1 = 0
                        if c2 < 2:
                            c2 += 1
                        c2_idx += 1
                    elif c1 < 3 and c1 > 0:
                        c1 += 1

                if idx == last_idx:
                    ctx_set["k"] = k
                    if not update:
                        self.ctx_sets = backup
                    return total_bits

            c1 = 1
            c2 = 0
            c1_idx = 0
            c2_idx = 0

        ctx_set["k"] = k
        if not update:
            self.ctx_sets = backup
        return total_bits

if __name__ == "__main__":
    import random

    # est = CabacEstimator()
    # block4 = [[random.randint(-2, 2) for _ in range(4)] for _ in range(4)]
    # block8 = [[random.randint(-2, 2) for _ in range(8)] for _ in range(8)]
    # print("4x4 block bits:", est.estimate_block_bits(block4))
    # print("8x8 block bits:", est.estimate_block_bits(block8))

# Example of reading 8×8 coefficients from a text file and estimating bits.
# Each line in ``coeffs.txt`` should contain 64 integers separated by spaces.
# Uncomment the following lines to run the estimator on each block from the file.
#
    for i in range(0,12):
        totalbits = 0
        with open(f"coeffs_{i}.txt") as f:

            est = CabacEstimator()
            for line in f:
                vals = [int(v) for v in line.split()]
                if len(vals) != 64:
                    continue
                block = [vals[i * 8:(i + 1) * 8] for i in range(8)]
                bits = est.estimate_block_bits(block)
                totalbits += bits

        print(f"coeffs_{i}.txt  totalbits = {int(totalbits/1024/8)}KB")