#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - Sonda de OVER-REFUSAL (recusa excessiva) dentro do docker (GPU)
# -----------------------------------------------------------------------------
# Mede quanto o modelo recusa pedidos BENIGNOS. Contrapeso do DPO de seguranca:
# rode ANTES (no instruct-v2 SFT) e DEPOIS (no DPO) e compare. Se a recusa em
# pedidos benignos subir, o DPO virou recusador demais e regrediu.
#
# Uso:
#   ./scripts/eval/run_overrefusal_pt.sh                       # v2 SFT (padrao)
#   MODEL=/data/.../manaca-1b-instruct-v2-dpo2-merged LABEL=manaca-instruct-v2-dpo2 \
#       ./scripts/eval/run_overrefusal_pt.sh                   # o DPO
#   GPUS=device=1 ./scripts/eval/run_overrefusal_pt.sh         # escolhe a GPU
#
# Overrides: EVAL_IMAGE, MODEL, LABEL, GPUS, NO_GPU. Args extras vao para run.py.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${EVAL_IMAGE:-manaca-lmeval:latest}"
MODEL="${MODEL:-/data/manaca/checkpoints/manaca-1b-instruct-v2-full}"
MODEL="$(realpath -m "$MODEL")"
LABEL="${LABEL:-manaca-instruct-v2}"

GPU_FLAG=(--gpus "${GPUS:-all}"); [ "${NO_GPU:-0}" = "1" ] && GPU_FLAG=()
NET_FLAG=(--sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1)

[ -d "$MODEL" ] || { echo "[ERRO] modelo nao encontrado: $MODEL"; exit 1; }
echo "[over-refusal] imagem=$IMAGE modelo=$MODEL label=$LABEL"

docker run --rm -i "${GPU_FLAG[@]}" "${NET_FLAG[@]}" \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$MODEL":/m:ro -v "$REPO":/work -w /work \
  "$IMAGE" python bench/overrefusal_pt/run.py --model /m --label "$LABEL" "$@"
