#!/usr/bin/env bash
# =============================================================================
# run_all_scenarios.sh
#
# Runs all four traffic scenarios against a vLLM endpoint (local or Parasail)
# and saves results to the results/ directory.
#
# Usage:
#   # Local vllm serve
#   bash scripts/run_all_scenarios.sh
#
#   # Parasail dedicated endpoint
#   ENDPOINT=https://api.parasail.io/v1 \
#   MODEL=xxxxxxxx-vllm-generative-dedicated \
#   bash scripts/run_all_scenarios.sh
# =============================================================================

set -euo pipefail

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
ENDPOINT="${ENDPOINT:-http://localhost:8000}"
USERS="${USERS:-20}"
MAX_TOKENS="${MAX_TOKENS:-256}"
DURATION="${DURATION:-60}"
RESULTS_DIR="${RESULTS_DIR:-results}"

PREFIX="parasail"
if [[ "$ENDPOINT" == *"localhost"* ]]; then
  PREFIX="local"
fi

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "  Running all scenarios"
echo "=============================================="
echo "  Model:    $MODEL"
echo "  Endpoint: $ENDPOINT"
echo "  Users:    $USERS"
echo "  Tokens:   $MAX_TOKENS"
echo "  Duration: ${DURATION}s each"
echo "  Results:  $RESULTS_DIR/"
echo "=============================================="
echo ""

TRAFFIC_GEN="traffic/vllm_traffic_gen.py"
if [[ ! -f "$TRAFFIC_GEN" ]]; then
  TRAFFIC_GEN="vllm_traffic_gen.py"
fi

run_scenario() {
  local SCENARIO=$1
  echo "▶ Running: $SCENARIO"
  python "$TRAFFIC_GEN" \
    --model "$MODEL" \
    --base-url "$ENDPOINT" \
    --scenario "$SCENARIO" \
    --users "$USERS" \
    --max-tokens "$MAX_TOKENS" \
    --duration "$DURATION" \
    --output-json "${RESULTS_DIR}/${PREFIX}_${SCENARIO}.json"
  echo "  ✓ Saved: ${RESULTS_DIR}/${PREFIX}_${SCENARIO}.json"
  echo ""
}

run_scenario steady
run_scenario burst
run_scenario ramp
run_scenario wave

echo "=============================================="
echo "  ✓ All scenarios complete"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/"
echo ""
echo "To visualize: open dashboard/index.html in your browser"
echo "and drop the JSON files onto the dashboard."
