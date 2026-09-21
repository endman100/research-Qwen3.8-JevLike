#!/usr/bin/env bash
set -euo pipefail
profile="${1:-6k71}"
case "$profile" in
  6k71) context=6144; sequences=71 ;;
  32k4) context=32768; sequences=4 ;;
  *) echo 'Usage: bash scripts/serve.sh {6k71|32k4}' >&2; exit 2 ;;
esac
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
model="${MODEL_PATH:-unsloth/Qwen3.8-27B-NVFP4}"
revision=()
if [[ -z "${MODEL_PATH:-}" ]]; then
  revision=(--revision f0b7c9e722f5565102fff8481c99e4d86ae099c7)
fi
# Keep local-only access; historical server used 0.0.0.0. No trust_remote_code.
exec vllm serve "$model" "${revision[@]}" \
  --served-model-name qwen3.8-27b --host 127.0.0.1 --port 8000 \
  --max-model-len "$context" --max-num-seqs "$sequences" \
  --max-num-batched-tokens 2048 --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --attention-backend TRITON_ATTN --enforce-eager \
  --safetensors-load-strategy eager
