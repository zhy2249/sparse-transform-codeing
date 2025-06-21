# Sparse Transform Coding for Video Residuals

该项目用于训练用于视频残差编码的稀疏变换字典（KSVD / MultiKSVD / IterMultiKSVD），并在多种方案下评估重构误差和熵编码比特数。

## 📦 文件说明

- `train_dicts.py`：训练字典
- `eval_with_dicts.py`：评估不同字典的重构性能与编码比特率
- `requirements.txt`：依赖库列表
- `.gitignore`：Git 忽略规则
- `LICENSE`：MIT 协议

## 📈 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
