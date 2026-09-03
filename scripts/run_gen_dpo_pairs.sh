#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - Gera pares de preferencia ON-POLICY para o DPO (dentro do docker)
# -----------------------------------------------------------------------------
# Uso:
#   MODE=safety      ./scripts/run_gen_dpo_pairs.sh        # regra (offline, GPU)
#   MODE=instruction ./scripts/run_gen_dpo_pairs.sh        # regra (offline, GPU)
#   MODE=quality ANTHROPIC_API_KEY=sk-ant-... ./scripts/run_gen_dpo_pairs.sh  # juiz (rede)
#
#   DETACH=1 MODE=safety ./scripts/run_gen_dpo_pairs.sh    # segundo plano + log
#
# Saida: data/dpo_onpolicy/<mode>.jsonl (NAO versionado; ver .gitignore).
# Overrides: MODEL, OUT, GPUS, EVAL_IMAGE, K, e args extras vao para o gen_dpo_pairs.py.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${EVAL_IMAGE:-manaca-lmeval:latest}"
MODE="${MODE:-safety}"
MODEL="${MODEL:-/data/manaca/checkpoints/manaca-1b-instruct-v2-full}"
MODEL="$(realpath -m "$MODEL")"
# Saida na raiz do repo (do usuario), NAO em data/ (que e do root). Fica dentro
# do mount do docker (/work) e coberta pelo .gitignore.
OUT="${OUT:-dpo_onpolicy/${MODE}.jsonl}"
K="${K:-4}"
OUTDIR_ABS="$REPO/$(dirname "$OUT")"

# Segundo plano: DETACH=1 re-executa destacado, com log.
if [ "${DETACH:-0}" = "1" ] && [ -z "${DPO_PAIRS_BG:-}" ]; then
  mkdir -p "$OUTDIR_ABS"
  LOG="$OUTDIR_ABS/gen_${MODE}_$(date +%Y%m%d_%H%M%S).log"
  DPO_PAIRS_BG=1 nohup "$0" "$@" > "$LOG" 2>&1 &
  echo "[dpo-pairs] SEGUNDO PLANO pid=$! | log: $LOG"
  echo "[dpo-pairs] acompanhar: tail -f '$LOG'"
  exit 0
fi

GPU_FLAG=(--gpus "${GPUS:-all}"); [ "${NO_GPU:-0}" = "1" ] && GPU_FLAG=()
NET_FLAG=(--sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1)

ENVX=(-e HF_HOME=/tmp/hf)
if [ "$MODE" = "quality" ]; then
  ENVX+=(-e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" -e JUDGE_API_KEY="${JUDGE_API_KEY:-}" \
         -e JUDGE_MODEL="${JUDGE_MODEL:-claude-opus-5}" -e JUDGE_PROVIDER="${JUDGE_PROVIDER:-}" \
         -e JUDGE_BASE_URL="${JUDGE_BASE_URL:-}")
else
  ENVX+=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)   # regra: offline
fi

[ -d "$MODEL" ] || { echo "[ERRO] modelo nao encontrado: $MODEL"; exit 1; }
mkdir -p "$OUTDIR_ABS"
echo "[dpo-pairs] modo=$MODE modelo=$MODEL out=$OUT"

docker run --rm -i "${GPU_FLAG[@]}" "${NET_FLAG[@]}" \
  --user "$(id -u):$(id -g)" "${ENVX[@]}" \
  -v "$MODEL":/m:ro -v "$REPO":/work -w /work \
  "$IMAGE" python dpo/gen_dpo_pairs.py --mode "$MODE" --model /m --out "$OUT" --k "$K" "$@"

echo "[dpo-pairs] pronto -> $REPO/$OUT"
