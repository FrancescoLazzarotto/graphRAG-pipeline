#!/usr/bin/env bash
# Multilingual sentence encoder for the cross-lingual vector channel — GPU 1.
#
# This is the half of retrieval that crosses the language gap: the graph is
# largely Italian, the gold questions are English, and lexical lookup cannot
# bridge that (exp_results/KG_VS_RETRIEVAL.md). Without this server the vector
# channel is unavailable and retrieval silently falls back to lexical-only.
#
# The command lived only inside run_campaign.sh's abort message, so every restart
# was retyped from memory and the August campaign ran three queries in three of
# six models without the channel. It is a script now.
#
# `--runner pooling` serves the model as an embedder rather than a generator.
# 0.12 memory utilisation leaves GPU 1 free for a generation server alongside it;
# max-model-len 512 is the e5 family's own limit.
#
# The index and the query encoder must use the SAME model and prefixes — see
# src/graphrag/embeddings.py. Changing MODEL here means rebuilding the index with
# scripts/kg_vector_index.py.
set -euo pipefail

MODEL="${GRAPHRAG_EMBED_MODEL:-intfloat/multilingual-e5-base}"
PORT="${EMBED_PORT:-8002}"
GPU="${EMBED_GPU:-1}"
UTIL="${EMBED_GPU_UTIL:-0.12}"

VLLM_BIN="${VLLM_BIN:-/mnt/storage/flazzarotto/venvs/vllm-serve/bin/vllm}"
export HF_HOME="${HF_HOME:-/mnt/storage/hf-cache}"

if curl -s --max-time 3 "http://localhost:${PORT}/v1/models" | grep -q '"id"'; then
  echo "encoder already serving on port ${PORT}"
  exit 0
fi

exec env CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
  --runner pooling \
  --port "$PORT" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len 512
