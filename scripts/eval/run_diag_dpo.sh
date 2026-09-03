#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - Diagnostico do DPO: base vs base+adapter vs merged (dentro do docker)
# -----------------------------------------------------------------------------
# Descarta o confounder do merge em bf16. Monta o dir de checkpoints e compara os
# tres modelos na mesma pergunta nociva (greedy).
#
# Uso:
#   ./scripts/eval/run_diag_dpo.sh                 # usa os caminhos padrao abaixo
#   CKPT_DIR=/data/manaca/checkpoints GPUS=device=0 ./scripts/eval/run_diag_dpo.sh
#
# Overrides: EVAL_IMAGE, CKPT_DIR, BASE, ADAPTER, MERGED (nomes relativos ao CKPT_DIR), GPUS.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${EVAL_IMAGE:-manaca-lmeval:latest}"
CKPT_DIR="${CKPT_DIR:-/data/manaca/checkpoints}"
CKPT_DIR="$(realpath -m "$CKPT_DIR")"
BASE="${BASE:-manaca-1b-instruct-v2-full}"
ADAPTER="${ADAPTER:-manaca-1b-instruct-v2-dpo2}"
MERGED="${MERGED:-manaca-1b-instruct-v2-dpo2-merged}"

GPU_FLAG=(--gpus "${GPUS:-all}"); [ "${NO_GPU:-0}" = "1" ] && GPU_FLAG=()
NET_FLAG=(--sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1)

[ -d "$CKPT_DIR/$BASE" ] || { echo "[ERRO] base nao encontrado: $CKPT_DIR/$BASE"; exit 1; }
echo "[diag] ckpt_dir=$CKPT_DIR base=$BASE adapter=$ADAPTER merged=$MERGED"

docker run --rm -i "${GPU_FLAG[@]}" "${NET_FLAG[@]}" \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$CKPT_DIR":/ckpt:ro -v "$REPO":/work -w /work \
  "$IMAGE" python dpo/diag_dpo.py \
    --base "/ckpt/$BASE" --adapter "/ckpt/$ADAPTER" --merged "/ckpt/$MERGED"
