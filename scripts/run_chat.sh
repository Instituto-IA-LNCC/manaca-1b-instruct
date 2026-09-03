#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - Chat interativo com um instruct, dentro do docker (GPU)
# -----------------------------------------------------------------------------
# Uso:
#   ./scripts/run_chat.sh                       # v2 (padrao), chat interativo
#   MODEL=/data/.../manaca-1b-instruct-full ./scripts/run_chat.sh   # v1
#   GPUS=device=1 ./scripts/run_chat.sh         # escolhe a GPU
#   ./scripts/run_chat.sh --greedy              # determinístico
#   ./scripts/run_chat.sh --prompt "sua pergunta aqui"   # uma pergunta só
#
# Overrides: EVAL_IMAGE, MODEL, GPUS, NO_GPU. Args extras vao para o chat.py.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${EVAL_IMAGE:-manaca-lmeval:latest}"
MODEL="${MODEL:-/data/manaca/checkpoints/manaca-1b-instruct-v2-full}"
MODEL="$(realpath -m "$MODEL")"

GPU_FLAG=(--gpus "${GPUS:-all}"); [ "${NO_GPU:-0}" = "1" ] && GPU_FLAG=()
NET_FLAG=(--sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1)

[ -d "$MODEL" ] || { echo "[ERRO] modelo nao encontrado: $MODEL"; exit 1; }
echo "[chat] imagem=$IMAGE modelo=$MODEL"

# -it: terminal interativo. Offline (modelo local). --user p/ nao criar root files.
docker run --rm -it "${GPU_FLAG[@]}" "${NET_FLAG[@]}" \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$MODEL":/m:ro -v "$REPO":/work -w /work \
  "$IMAGE" python scripts/chat.py --model /m "$@"
