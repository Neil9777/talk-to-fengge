#!/bin/bash
# /workspace/setup.sh — wheel 缓存版
# 首次运行：~10-12 分钟（flash_attn 从源码编译）
# 后续运行：~2 分钟（直接装本地 wheel，跳过编译）
set -e
LOG=/workspace/setup.log
WHEEL_DIR=/workspace/wheels
mkdir -p "$WHEEL_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "[setup] 开始 $(date)"

# 1. Torch + CUDA 12.8
echo "[setup] 安装 torch 2.7.0+cu128..."
pip install --force-reinstall torch==2.7.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128 -q

# 2. nano_vllm_voxcpm
echo "[setup] 安装 nano_vllm_voxcpm==2.0.2..."
pip install nano_vllm_voxcpm==2.0.2 --no-cache-dir -q

# 3. flash_attn — 优先用本地 wheel，没有才编译并存盘
FLASH_WHL=$(ls "$WHEEL_DIR"/flash_attn*.whl 2>/dev/null | head -1)
if [ -n "$FLASH_WHL" ]; then
    echo "[setup] 使用缓存 wheel: $(basename $FLASH_WHL)（跳过编译）"
    pip install "$FLASH_WHL" -q
else
    echo "[setup] 首次编译 flash_attn（约 8 分钟，只需做一次）..."
    pip uninstall flash-attn -y 2>/dev/null || true
    # 编译并同时保存 wheel 到 /workspace/wheels/
    MAX_JOBS=4 pip wheel flash-attn --no-build-isolation --no-cache-dir \
        -w "$WHEEL_DIR" -q
    FLASH_WHL=$(ls "$WHEEL_DIR"/flash_attn*.whl 2>/dev/null | head -1)
    if [ -n "$FLASH_WHL" ]; then
        pip install "$FLASH_WHL" -q
        echo "[setup] wheel 已保存: $(basename $FLASH_WHL)（下次迁移直接用）"
    else
        echo "[setup] wheel 保存失败，尝试直接安装..."
        MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-cache-dir -q
    fi
fi

# 4. 其他依赖
pip install -q fastapi "uvicorn[standard]" soundfile numpy huggingface_hub

# 5. 下载模型权重（如需）
if [ ! -f /workspace/voxcpm2_weights/config.json ]; then
    echo "[setup] 下载 voxcpm2 权重..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('openbmb/VoxCPM2', local_dir='/workspace/voxcpm2_weights')
print('[setup] 权重下载完成')
"
fi

# 6. 验证
echo "[setup] 验证..."
python3 -c "
import torch; print(f'torch {torch.__version__}, CUDA={torch.cuda.is_available()}')
import nanovllm_voxcpm; print(f'nanovllm_voxcpm {nanovllm_voxcpm.__version__}')
import flash_attn; print(f'flash_attn {flash_attn.__version__}')
"
echo "[setup] 完成 $(date)"
