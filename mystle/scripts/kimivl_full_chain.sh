#!/bin/bash
set -e

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export HF_HOME=$HOME/fsas/models/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export PYTHONPATH=/root/DeepSeek-VL2
export HF_DATASETS_CACHE=$HOME/fsas/datasets/bridge-vlm/hf_datasets_cache
export LMMS_EVAL_DATASETS_CACHE=$HF_DATASETS_CACHE

cd /root/DeepSeek-VL2
source .venv/bin/activate

RESULTS_ROOT=$HOME/fsas/projects/bridge-vlm/results/kimivl
CALIB_DIR=$HOME/fsas/datasets/deepseek-vl2-bridge/calibration
MODEL_PATH=moonshotai/Kimi-VL-A3B-Thinking
EVAL_SCRIPT=mystle/experiments/kimivl_evaluate.py
HF_CACHE=$HF_HOME/hub/models--moonshotai--Kimi-VL-A3B-Thinking

mkdir -p $RESULTS_ROOT/logs

echo "===== $(date) ===== Kimi-VL Full Pipeline Start ====="

# ===== Wait for model download (robust check) =====
echo "===== $(date) ===== Waiting for model download ====="
MAX_WAIT=7200
ELAPSED=0
while true; do
    BLOB_SIZE=$(du -sm $HF_CACHE/blobs/ 2>/dev/null | cut -f1)
    INCOMPLETE=$(find $HF_CACHE -name '*.incomplete' 2>/dev/null | wc -l)
    echo "$(date) Blobs: ${BLOB_SIZE:-0}MB, Incomplete files: $INCOMPLETE"
    
    if [ "$INCOMPLETE" -eq 0 ] && [ "${BLOB_SIZE:-0}" -gt 30000 ]; then
        echo "$(date) No incomplete files and blobs > 30GB — checking model loadability..."
        if python -c "
from transformers import AutoProcessor, AutoConfig
proc = AutoProcessor.from_pretrained('$MODEL_PATH', trust_remote_code=True)
cfg = AutoConfig.from_pretrained('$MODEL_PATH', trust_remote_code=True)
print(f'Processor + Config loaded OK. model_type={cfg.model_type}')
" 2>/dev/null; then
            echo "$(date) Model files are ready!"
            break
        else
            echo "$(date) Processor/Config load failed, continuing to wait..."
        fi
    fi

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Model download timeout after ${MAX_WAIT}s"
        exit 1
    fi
    sleep 60
    ELAPSED=$((ELAPSED + 60))
done

# ===== Verify full model load =====
echo "===== $(date) ===== Verifying model loadability ====="
python -c "
from transformers import AutoModelForCausalLM
import torch
print('Loading model on CPU to verify...')
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_PATH', trust_remote_code=True, torch_dtype=torch.bfloat16,
)
print(f'Model loaded: {type(model).__name__}')
n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f'Parameters: {n_params:.2f}B')
lm = model.language_model if hasattr(model, 'language_model') else model
if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
    for i, layer in enumerate(lm.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'experts'):
            n_exp = len(layer.mlp.experts)
            print(f'MoE layer {i}: {n_exp} experts')
            break
del model
import gc; gc.collect()
print('Model verification passed!')
"

echo "===== $(date) ===== Model verified. Starting experiments ====="

# ===== C6: Bridge Score =====
echo "===== $(date) ===== C6: Bridge Score ====="
BRIDGE_SUMMARY=$RESULTS_ROOT/bridge_score/layer_bridge_summary.json
if [ -f "$BRIDGE_SUMMARY" ]; then
    echo "C6 bridge score already exists, skipping."
else
    python mystle/experiments/kimivl_bridge_score.py \
        --model-path $MODEL_PATH \
        --calib-manifest $CALIB_DIR/manifest.json \
        --output-dir $RESULTS_ROOT/bridge_score \
        --num-samples 128 \
        2>&1 | tee $RESULTS_ROOT/logs/c6_bridge_score.log
fi
BRIDGE_SUMMARY=$RESULTS_ROOT/bridge_score/layer_bridge_summary.json

# ===== C7: HC-SMoE 50% merge =====
echo "===== $(date) ===== C7: HC-SMoE 50% merge ====="
C7_DIR=$RESULTS_ROOT/baselines/hcsmoe_keep_0p5
if [ -f "$C7_DIR/merged_model/config.json" ]; then
    echo "C7 already done, skipping."
else
    mkdir -p $C7_DIR
    python mystle/experiments/kimivl_cpu_merge.py \
        --model-path $MODEL_PATH \
        --compression-keep 0.5 \
        --output-dir $C7_DIR/merged_model \
        2>&1 | tee $RESULTS_ROOT/logs/c7_hcsmoe_merge.log
fi

# ===== C8: HC-SMoE+Bridge 50% merge =====
echo "===== $(date) ===== C8: Admissibility 50% merge ====="
C8_DIR=$RESULTS_ROOT/baselines/admissibility_keep_0p5
if [ -f "$C8_DIR/merged_model/config.json" ]; then
    echo "C8 already done, skipping."
else
    mkdir -p $C8_DIR
    python mystle/experiments/kimivl_cpu_merge.py \
        --model-path $MODEL_PATH \
        --compression-keep 0.5 \
        --output-dir $C8_DIR/merged_model \
        --layer-bridge-summary $BRIDGE_SUMMARY \
        2>&1 | tee $RESULTS_ROOT/logs/c8_admissibility_merge.log
fi

# ===== D6: Kimi-VL Base eval =====
echo "===== $(date) ===== D6: Kimi-VL Base eval ====="
D6_MM=$RESULTS_ROOT/d6_base_mm
D6_TEXT=$RESULTS_ROOT/d6_base_text
if [ -d "$D6_MM" ] && [ -d "$D6_TEXT" ]; then
    echo "D6 already done, skipping."
else
    python $EVAL_SCRIPT mm \
        --model-path $MODEL_PATH \
        --output-dir $D6_MM \
        2>&1 | tee $RESULTS_ROOT/logs/d6_base_mm.log
    python $EVAL_SCRIPT text \
        --model-path $MODEL_PATH \
        --output-dir $D6_TEXT \
        2>&1 | tee $RESULTS_ROOT/logs/d6_base_text.log
fi

# ===== D7: HC-SMoE 50% eval =====
echo "===== $(date) ===== D7: HC-SMoE 50% eval ====="
D7_MM=$RESULTS_ROOT/d7_hcsmoe_0p5_mm
D7_TEXT=$RESULTS_ROOT/d7_hcsmoe_0p5_text
if [ -d "$D7_MM" ] && [ -d "$D7_TEXT" ]; then
    echo "D7 already done, skipping."
else
    python $EVAL_SCRIPT mm \
        --model-path $C7_DIR/merged_model \
        --output-dir $D7_MM \
        2>&1 | tee $RESULTS_ROOT/logs/d7_hcsmoe_mm.log
    python $EVAL_SCRIPT text \
        --model-path $C7_DIR/merged_model \
        --output-dir $D7_TEXT \
        2>&1 | tee $RESULTS_ROOT/logs/d7_hcsmoe_text.log
fi

# ===== D8: Admissibility 50% eval =====
echo "===== $(date) ===== D8: Admissibility 50% eval ====="
D8_MM=$RESULTS_ROOT/d8_admissibility_0p5_mm
D8_TEXT=$RESULTS_ROOT/d8_admissibility_0p5_text
if [ -d "$D8_MM" ] && [ -d "$D8_TEXT" ]; then
    echo "D8 already done, skipping."
else
    python $EVAL_SCRIPT mm \
        --model-path $C8_DIR/merged_model \
        --output-dir $D8_MM \
        2>&1 | tee $RESULTS_ROOT/logs/d8_admissibility_mm.log
    python $EVAL_SCRIPT text \
        --model-path $C8_DIR/merged_model \
        --output-dir $D8_TEXT \
        2>&1 | tee $RESULTS_ROOT/logs/d8_admissibility_text.log
fi

echo "===== $(date) ===== ALL KIMI-VL EXPERIMENTS COMPLETE ====="
echo "Results in: $RESULTS_ROOT"
ls -la $RESULTS_ROOT/
