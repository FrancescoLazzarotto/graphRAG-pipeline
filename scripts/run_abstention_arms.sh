#!/usr/bin/env bash
# Three arms measuring the abstention path, all in one server session.
#
# The thesis campaigns E1-E8 ran against an answer prompt whose closing line
# permitted a declaration of insufficiency only for an empty context. That
# wording, and the absence of a terminal state producing a refusal, left the
# system abstaining on 11 of 192 distractor cells. Both were repaired after the
# campaigns closed and neither repair is measured on the reference set.
#
# The arms differ in the abstention mechanism and in nothing else:
#   A0  pre-repair wording, no gate      reproduces the E6 configuration
#   A1  repaired wording, no gate        isolates the prompt line
#   A2  repaired wording, domain gate    adds the terminal refusal state
#
# One server session throughout, so the comparison is within-session and the
# +/-0.03 cross-session band does not apply.
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_ROOT="${OUT_ROOT:-/srv/projects/graphllm/experiments/exp_results_abstention}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
GOLD="${GOLD:-gold_v3.json}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
STRATEGIES="default,hybrid,text_only,no_retrieval,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path"

mkdir -p "$OUT_ROOT"

preflight() {
  curl -sf --max-time 10 "$BASE_URL/models" > /dev/null \
    || { echo "generator not answering at $BASE_URL"; exit 1; }
  curl -sf --max-time 10 http://localhost:8002/v1/models > /dev/null \
    || { echo "embedding encoder not answering on 8002"; exit 1; }
  # Counting carriers is not enough. A reload of the graph store reassigns every
  # internal identifier, which leaves the carriers in place and pointing at
  # nothing: the count still passes, the vector channel silently degrades to
  # lexical matching, and the campaign looks complete. Measured once, that cost
  # 0.03 to 0.06 concept F1 on every graph strategy and nothing in the log said
  # so. Check that the identifiers still resolve.
  conda run -n graphllm python scripts/check_vector_index.py --min-resolving 1000 \
    || { echo "vector index unusable; rebuild with scripts/kg_vector_index.py"; exit 1; }
  echo "preflight ok: generator, encoder and a resolving vector index"
}

run_arm() {
  local tag="$1"; shift
  echo "=== arm ${tag} : $(date -Is) ==="
  conda run --no-capture-output -n graphllm python -m graphrag.cli --experiment \
    --questions-file "$GOLD" \
    --strategies "$STRATEGIES" \
    --llm --vllm --vllm-base-url "$BASE_URL" \
    --model-id "$MODEL" \
    --max-new-tokens 1024 --max-context-tokens 6000 \
    --complexity medium --enforce-language \
    --cite-evidence --citation-policy mark --citation-display label \
    --prefer-verbatim-definitions \
    --text-docs-dir artifacts/corpus_circular22 --text-retriever-backend tfidf \
    --text-retriever-mmr --text-retriever-mmr-lambda 0.7 \
    --text-retriever-max-per-doc 2 \
    --vector-retrieval --seed-from-retrieved --subgraph-seed-count 3 \
    --focused-answer --evidence-max-triple-items "${EVIDENCE_CAP:-30}" \
    --output-dir "${OUT_ROOT}/${tag}" \
    --experiment-tag "abst_${tag}" \
    "$@"
  echo "=== arm ${tag} done : $(date -Is) ==="
}

preflight
run_arm a0_legacy_prompt --legacy-insufficiency-wording
run_arm a1_repaired_prompt
run_arm a2_repaired_plus_gate --enable-domain-gate
echo "ALL ARMS COMPLETE $(date -Is)"
