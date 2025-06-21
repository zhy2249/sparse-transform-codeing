# 视频残差稀疏变换编码工具

本项目提供了一套脚本，用于训练稀疏字典以压缩视频残差，并评估这些字典在不同量化条件下的表现。核心思路是通过学习型稀疏变换提升传统 DCT 与 VVC MTS 在残差编码中的效率。

## 目录
- `train_dicts.py`：字典训练脚本，支持 KSVD、MultiKSVD 和基于 RD 的 IterMultiKSVD。
- `eval_with_dicts.py`：加载训练好的字典（或 DCT、VVC MTS）对测试序列进行变换编码，记录 PSNR 与比特率。
- `CABAC_Estimator.py`：简化版的 HEVC CABAC 比特估计器。
- `TU4.cfg`、`TU8.cfg`：外部 CABAC 编码器的示例配置文件。
- `resVideo/`：示例 YUV 残差序列。
- `requirements.txt`：依赖库列表。

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练字典
```bash
python train_dicts.py \
  --train_video resVideo/train.yuv \
  --widths 1920 --heights 1080 \
  --block 8 --sparsity 6 --iter 20 \
  --output_dir trained_dicts
```

### 评估字典
```bash
python eval_with_dicts.py \
  --test_video resVideo/test.yuv \
  --width 1920 --height 1080 \
  --dict_dir trained_dicts \
  --qp 32
```

完整参数和更多功能请使用 `-h` 查看帮助。

## 许可协议
本仓库暂未包含正式的 LICENSE 文件，默认遵循 MIT 风格的开源条款。
