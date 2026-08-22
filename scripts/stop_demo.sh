#!/usr/bin/env bash
# Stop everything scripts/start_demo.sh started, in reverse order.
#
# Kills by the pid files that script wrote, then verifies the ports actually
# went quiet: a vLLM server ignores SIGTERM while it is loading weights, and a
# half-dead one holds its GPU memory, so the next start fails with an
# out-of-memory that looks like a different problem entirely.
#
#   scripts/stop_demo.sh            # everything
#   scripts/stop_demo.sh encoder    # one component, by its start_demo label
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${DEMO_LOG_DIR_RUNTIME:-$ROOT/artifacts/demo_logs}"
GRACE_SEC="${DEMO_STOP_GRACE_SEC:-30}"

if [[ ! -d "$LOG_DIR" ]]; then
  echo "Niente da fermare: $LOG_DIR non esiste."
  exit 0
fi

targets=()
if [[ $# -gt 0 ]]; then
  targets=("$@")
else
  for pid_file in "$LOG_DIR"/*.pid; do
    [[ -e "$pid_file" ]] || continue
    targets+=("$(basename "$pid_file" .pid)")
  done
fi

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "Nessun processo registrato in $LOG_DIR."
  exit 0
fi

for label in "${targets[@]}"; do
  pid_file="$LOG_DIR/$label.pid"
  if [[ ! -f "$pid_file" ]]; then
    echo "  $label: nessun pid registrato"
    continue
  fi
  pid="$(cat "$pid_file")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "  $label: già fermo"
    rm -f "$pid_file"
    continue
  fi
  # The pid is the wrapper script; the server is its child, so the whole
  # process group goes down together.
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  waited=0
  while kill -0 "$pid" 2>/dev/null && (( waited < GRACE_SEC )); do
    sleep 1
    waited=$((waited + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "  $label: non è uscito in ${GRACE_SEC}s, SIGKILL"
    kill -KILL -- "-$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  else
    echo "  $label: fermato"
  fi
  rm -f "$pid_file"
done

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "GPU dopo lo stop:"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
fi
