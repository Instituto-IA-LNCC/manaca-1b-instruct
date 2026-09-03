#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - Monta o dataset de SAFETY-SFT (dentro do docker, GPU p/ benign-help)
# -----------------------------------------------------------------------------
# Gera: recusar o nocivo + ajudar o benigno (on-policy) + dados gerais.
# Saida (output = recusa OU resposta util; SEM conteudo nocivo) -> versionavel.
#
# Uso:
#   ./scripts/run_gen_safety_sft.sh                          # padrao (com benign-help + geral)
#   N_GENERAL=400 K_REFUSE=3 ./scripts/run_gen_safety_sft.sh
#   NO_GENERAL=1 ./scripts/run_gen_safety_sft.sh             # sem misturar dados gerais
#
# Overrides: EVAL_IMAGE, MODEL, OUT, GENERAL_DATA, N_GENERAL, K_REFUSE, GPUS.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${EVAL_IMAGE:-manaca-lmeval:latest}"
MODEL="${MODEL:-/data/manaca/checkpoints/manaca-1b-instruct-v2-full}"
MODEL="$(realpath -m "$MODEL")"
OUT="${OUT:-sft_safety/safety_sft.jsonl}"
# Receita vencedora (safe4): K_REFUSE=4 recusas por prompt + benign-help disjunto
# + 200 gerais. Ver docs/evaluation/safety-alignment-pt.md.
K_REFUSE="${K_REFUSE:-4}"
N_GENERAL="${N_GENERAL:-200}"
# Dados gerais do SFT v2 (root-owned em data/; montado read-only). Vazio/NO_GENERAL desliga.
GENERAL_DATA="${GENERAL_DATA:-data/sft_v2/manaca_sft.jsonl}"
OUTDIR_ABS="$REPO/$(dirname "$OUT")"

GPU_FLAG=(--gpus "${GPUS:-all}"); [ "${NO_GPU:-0}" = "1" ] && GPU_FLAG=()
NET_FLAG=(--sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1)

[ -d "$MODEL" ] || { echo "[ERRO] modelo nao encontrado: $MODEL"; exit 1; }
mkdir -p "$OUTDIR_ABS"

GEN_ARG=()
if [ "${NO_GENERAL:-0}" != "1" ] && [ -n "$GENERAL_DATA" ] && [ -f "$REPO/$GENERAL_DATA" ]; then
  GEN_ARG=(--general_data "/work/$GENERAL_DATA" --n_general "$N_GENERAL")
else
  echo "[safety-sft] AVISO: sem dados gerais ($GENERAL_DATA) -> maior risco de over-refusal"
fi

echo "[safety-sft] modelo=$MODEL out=$OUT k_refuse=$K_REFUSE n_general=$N_GENERAL"
docker run --rm -i "${GPU_FLAG[@]}" "${NET_FLAG[@]}" \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$MODEL":/m:ro -v "$REPO":/work -w /work \
  "$IMAGE" python sft/gen_safety_sft.py --model /m --out "$OUT" \
    --k_refuse "$K_REFUSE" "${GEN_ARG[@]}"

echo "[safety-sft] pronto -> $REPO/$OUT"
