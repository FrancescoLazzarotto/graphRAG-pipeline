#!/usr/bin/env bash
# Avvio vLLM ottimizzato per 1x NVIDIA A40 46GB
#
# Flag chiave:
#   --enable-prefix-caching   RadixAttention: riusa KV cache per prefissi condivisi
#                             (system prompt + relation vocab ripetuti ogni chunk = speedup ~20-40%)
#   --enable-chunked-prefill  Riduce latency spikes su prompt lunghi
#   --max-num-seqs 16         Continuous batching fino a 16 request parallele
#   --gpu-memory-utilization  Lascia ~6GB per GLiNER + SentenceTransformer + CUDA overhead
#
# Speculative decoding (opzionale, ~2-3x speedup su generation, costa ~3-4GB VRAM):
#   Decommentare le righe --speculative-model / --num-speculative-tokens
#   e abbassare --gpu-memory-utilization a 0.83

MODEL="${VLLM_MODEL_NAME:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
PORT="${VLLM_PORT:-8000}"

exec vllm serve "$MODEL" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.87 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --dtype float16 \
  --port "$PORT"
  # --speculative-model Qwen/Qwen2.5-1.5B-Instruct \
  # --num-speculative-tokens 5
