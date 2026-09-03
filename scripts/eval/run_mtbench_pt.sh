#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - MT-Bench-PT: geracao das respostas de UM instruct dentro do docker
# -----------------------------------------------------------------------------
# Dois modos:
#   LOCAL (modelo em disco, ex.: o Manaca):
#     MODEL=/data/brunolsm/manaca-checkpoints/manaca-1b-instruct-full \
#     LABEL=manaca-instruct-v1 ./scripts/eval/run_mtbench_pt.sh
#
#   HUB (baixa do Hugging Face, ex.: baselines instruct):
#     HFID=TucanoBR/Tucano-1b1-Instruct LABEL=tucano-1b1-instruct \
#       ./scripts/eval/run_mtbench_pt.sh
#
# JUSTICA: cada modelo e interrogado com o SEU proprio chat_template
# (--prompt_style auto). O Manaca, que nao tem chat_template, cai no Alpaca-PT do
# SFT. Nao force um template unico entre modelos diferentes.
#
# Overrides: EVAL_IMAGE, MAX_NEW_TOKENS, ATTN, HF_CACHE, HF_TOKEN, PROMPT_STYLE, NO_GPU.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${EVAL_IMAGE:-manaca-lmeval:latest}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
ATTN="${ATTN:-sdpa}"
PROMPT_STYLE="${PROMPT_STYLE:-auto}"
# Conjunto de perguntas + pasta de saida (default: MT-Bench-PT). Para o IFEval-PT:
#   QUESTIONS=bench/ifeval_pt/prompts.jsonl ANS_DIR=bench/ifeval_pt/answers
QUESTIONS="${QUESTIONS:-bench/mtbench_pt/questions.jsonl}"
ANS_DIR="${ANS_DIR:-bench/mtbench_pt/answers}"

GPU_FLAG=(--gpus all); [ "${NO_GPU:-0}" = "1" ] && GPU_FLAG=()
NET_FLAG=(--sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1)
[ "${NO_IP4:-0}" = "1" ] && NET_FLAG=()

mkdir -p "$REPO/$ANS_DIR"

# Monta os args do docker conforme o modo (local vs hub).
MODEL_MOUNT=(); CACHE_MOUNT=(); ENV_FLAGS=()
if [ -n "${HFID:-}" ]; then
  MODEL_ARG="$HFID"
  LABEL="${LABEL:-$(basename "$HFID")}"
  # Cache PROPRIO (do usuario). Nao reuse $HOME/hf_cache_eval: ele foi populado
  # pelas rodadas do lm-eval que rodavam docker como root, e com --user (abaixo) o
  # processo nao consegue escrever la (PermissionError).
  HF_CACHE="${HF_CACHE:-$HOME/hf_cache_mtbench}"; HF_CACHE="$(realpath -m "$HF_CACHE")"; mkdir -p "$HF_CACHE"
  CACHE_MOUNT=(-v "$HF_CACHE":/hf)
  ENV_FLAGS=(-e HF_HOME=/hf -e HF_TOKEN="${HF_TOKEN:-}")   # online: baixa do HF
  echo "[mtbench] modo=HUB  hfid=$HFID  rotulo=$LABEL  cache=$HF_CACHE"
else
  MODEL="${MODEL:?defina MODEL=/caminho (local) ou HFID=org/model (hub)}"
  MODEL="$(realpath -m "$MODEL")"
  MODEL_ARG="/m"
  LABEL="${LABEL:-$(basename "$MODEL")}"
  MODEL_MOUNT=(-v "$MODEL":/m:ro)
  ENV_FLAGS=(-e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)  # offline
  echo "[mtbench] modo=LOCAL  modelo=$MODEL  rotulo=$LABEL"
fi

OUT="$ANS_DIR/${LABEL}.jsonl"

docker run --rm -i "${GPU_FLAG[@]}" "${NET_FLAG[@]}" \
  --user "$(id -u):$(id -g)" \
  "${ENV_FLAGS[@]}" \
  "${MODEL_MOUNT[@]}" "${CACHE_MOUNT[@]}" \
  -v "$REPO":/work -w /work \
  "$IMAGE" python bench/mtbench_pt/gen_answers.py \
    --model "$MODEL_ARG" --model_label "$LABEL" \
    --questions "$QUESTIONS" \
    --out "$OUT" --max_new_tokens "$MAX_NEW_TOKENS" --attn "$ATTN" \
    --prompt_style "$PROMPT_STYLE"

echo "[mtbench] respostas -> $REPO/$OUT"
echo "[mtbench] proximo passo: rodar o juiz (bench/mtbench_pt/judge.py). Veja o README."
