#!/usr/bin/env bash
# =============================================================================
# Manacá — Pós-treino SFT/DPO (Fase 3) em Docker  [substitui o SLURM de references/phase3-sft §57.5]
# =============================================================================
# Equivalências:
#   module load cuda + conda activate manaca-sft  ->  imagem manaca-posttrain
#   accelerate launch ... (sob srun)              ->  docker run ... accelerate launch
#   Singularity .sif                              ->  imagem Docker
#
# Uso:
#   make build-posttrain
#   ./scripts/docker/run_sft.sh sft  --model_name_or_path menezesbruno/manaca-1b-base ...
#   ./scripts/docker/run_sft.sh dpo  --model_name_or_path .../manaca-1b-instruct2 ...
#
# O primeiro argumento é o modo (sft|dpo); os demais vão para o script de treino.
# Os scripts sft/train.py e dpo/train.py são artefatos da Fase 3 (crie-os no
# repositório espelhando llm-jp-sft / llm-jp-dpo com a API moderna do trl).
# =============================================================================
set -euo pipefail

MODE="${1:-sft}"; shift || true
case "${MODE}" in
  sft) TRAIN_SCRIPT="${SFT_SCRIPT:-sft/train.py}" ;;
  dpo) TRAIN_SCRIPT="${DPO_SCRIPT:-dpo/train.py}" ;;
  *) echo "Modo invalido: '${MODE}' (use: sft | dpo)"; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Carrega .env como DEFAULTS — ambiente/CLI têm precedência sobre o .env.
load_env_defaults() {
    [ -f "$1" ] || return 0
    local line key
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|\#*) continue ;; esac
        key=${line%%=*}; key=${key// /}
        case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac
        [ -z "${!key+x}" ] && export "${key}=${line#*=}"
    done < "$1"
}
load_env_defaults .env

IMAGE="${POSTTRAIN_IMAGE:-manaca-posttrain:latest}"
DATA_DIR="$(realpath -m "${DATA_DIR:-./data}")"
HF_CACHE_DIR="$(realpath -m "${HF_CACHE_DIR:-./hf-cache}")"
CKPT_DIR="$(realpath -m "${CKPT_DIR:-./checkpoints}")"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_config_zero3.yaml}"

mkdir -p "${DATA_DIR}" "${HF_CACHE_DIR}" "${CKPT_DIR}"

# Fail-fast: script de treino e config accelerate precisam existir no repo.
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "[run_sft] ERRO: script de treino ausente: ${TRAIN_SCRIPT}"
    echo "         Crie-o (espelhando llm-jp-sft/llm-jp-dpo com a API moderna do trl:"
    echo "         SFTConfig/DPOConfig) ou aponte via SFT_SCRIPT=/DPO_SCRIPT=."
    exit 1
fi
if [ ! -f "${ACCELERATE_CONFIG}" ]; then
    echo "[run_sft] ERRO: config accelerate ausente: ${ACCELERATE_CONFIG}"
    echo "         Ajuste ACCELERATE_CONFIG=<arquivo> (ex.: configs/accelerate_config_zero3.yaml)."
    exit 1
fi

echo "[run_sft] modo=${MODE} script=${TRAIN_SCRIPT} gpus=${GPUS_PER_NODE} config=${ACCELERATE_CONFIG}"

# O accelerate 1.1.x NÃO aceita 'auto' em gradient_accumulation_steps/gradient_clipping
# no YAML; e o valor precisa CASAR com o do SFTConfig (senão o transformers acusa
# mismatch). Extraímos o --gradient_accumulation_steps de "$@" (default 16 = default
# do train.py) e geramos um config com os valores concretos.
GAS=16; prev=""
for a in "$@"; do
  case "$a" in --gradient_accumulation_steps=*) GAS="${a#*=}" ;; esac
  [ "$prev" = "--gradient_accumulation_steps" ] && GAS="$a"
  prev="$a"
done
LAUNCH_CFG="configs/.accelerate_${MODE}.gen.yaml"
sed -e "s/gradient_accumulation_steps: auto/gradient_accumulation_steps: ${GAS}/" \
    -e "s/gradient_clipping: auto/gradient_clipping: 1.0/" \
    "${ACCELERATE_CONFIG}" > "${LAUNCH_CFG}"
echo "[run_sft] accelerate config gerado: ${LAUNCH_CFG} (gradient_accumulation_steps=${GAS})"

# Porta do rendezvous do torch.distributed. Como usamos --network host, a porta é
# do HOST: dois treinos simultâneos colidem na 29500 (EADDRINUSE) e um morre na
# inicialização. Damos uma porta ÚNICA por run (override: MANACA_MASTER_PORT).
MAIN_PORT="${MANACA_MASTER_PORT:-$((29500 + RANDOM % 400))}"
echo "[run_sft] main_process_port=${MAIN_PORT} (evita colisão de porta entre runs)"

# ── Log em arquivo (mesmo padrão do run_pretrain.sh) ─────────────────────────
# Salva TUDO do treino (stdout+stderr) num .log timestamped + proveniência, no
# CKPT_DIR (disco montado) — para o registro científico e reprodutibilidade.
LOG_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${CKPT_DIR}/${MODE}-logs"
LOG_FILE="${RUN_DIR}/${MODE}_${LOG_TS}.log"
PROV_FILE="${RUN_DIR}/provenance_${MODE}_${LOG_TS}.txt"
mkdir -p "${RUN_DIR}"

GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo '?')"
GIT_BR="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
{
  echo "Manaca ${MODE} — proveniencia do run"
  echo "Timestamp:  ${LOG_TS}"
  echo "Git:        ${GIT_SHA} (${GIT_BR})"
  echo "Imagem:     ${IMAGE}"
  echo "GPUs:       ${GPUS_PER_NODE}"
  echo "Accelerate: ${ACCELERATE_CONFIG}"
  echo "Port:       ${MAIN_PORT}"
  echo "Script:     ${TRAIN_SCRIPT}"
  echo "Args:       $*"
} | tee "${PROV_FILE}"

echo "[run_sft] log: ${LOG_FILE}"

# DNS: o Docker injeta no container um DNS que não resolve neste cluster. Geramos
# um resolv.conf com o(s) servidor(es) DNS REAL(is) do host (via resolvectl) +
# 127.0.0.53 de fallback + single-request-reopen (corrige a resolução A/AAAA
# intermitente do glibc), para o treino resolver o HuggingFace.
# Override: MANACA_DNS="ns1 ns2".
RESOLV_MOUNT=()
RESOLV_CONF="$(realpath -m "${CKPT_DIR}/.manaca-resolv.conf")"
if { if [ -n "${MANACA_DNS:-}" ]; then
        for ns in ${MANACA_DNS}; do echo "nameserver ${ns}"; done
     else
        resolvectl dns 2>/dev/null \
          | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}|([0-9A-Fa-f]{1,4}:){2,}[0-9A-Fa-f:]*' \
          | grep -vE '^(127\.|::1$)' | sort -u | sed 's/^/nameserver /'
        echo 'nameserver 127.0.0.53'
     fi
     echo 'options single-request-reopen timeout:2 attempts:5'; } > "${RESOLV_CONF}" 2>/dev/null; then
  RESOLV_MOUNT=(-v "${RESOLV_CONF}:/etc/resolv.conf:ro")
fi

# Argumentos comuns do docker run (iguais no primeiro e no segundo plano).
COMMON_ARGS=(--gpus all --network host --ipc host
  --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864
  "${RESOLV_MOUNT[@]}"
  -e WORK_DIR=/workspace/manaca-corpus
  -e HF_HOME=/workspace/hf-cache
  -e HF_TOKEN="${HF_TOKEN:-}"
  -e WANDB_API_KEY="${WANDB_API_KEY:-}"
  -e WANDB_PROJECT="${WANDB_PROJECT:-manaca-${MODE}}"
  -v "${REPO_ROOT}":/workspace/manaca
  -v "${DATA_DIR}":/workspace/manaca-corpus
  -v "${HF_CACHE_DIR}":/workspace/hf-cache
  -v "${CKPT_DIR}":/workspace/checkpoints
  -w /workspace/manaca)

if [ "${DETACH:-0}" = "1" ] || [ "${RUN_BG:-0}" = "1" ]; then
  # ── Segundo plano: container destacado (sobrevive ao logout do SSH) ─────────
  # O log é escrito DENTRO do container, no volume montado -> persiste no host em
  # ${LOG_FILE}. printf %q garante quoting seguro dos argumentos de treino.
  CNAME="manaca-${MODE}-${LOG_TS}"
  CLOG="/workspace/checkpoints/${MODE}-logs/${MODE}_${LOG_TS}.log"
  INNER="accelerate launch --config_file $(printf %q "${LAUNCH_CFG}") --main_process_port $(printf %q "${MAIN_PORT}") --num_processes $(printf %q "${GPUS_PER_NODE}") $(printf %q "${TRAIN_SCRIPT}")"
  for a in "$@"; do INNER="${INNER} $(printf %q "$a")"; done
  INNER="${INNER} 2>&1 | tee -a $(printf %q "${CLOG}")"
  cid="$(docker run -d --name "${CNAME}" "${COMMON_ARGS[@]}" "${IMAGE}" bash -c "${INNER}")"
  echo "[run_sft] SEGUNDO PLANO | detached — container=${CNAME}"
  echo "[run_sft]   id: ${cid}"
  echo "[run_sft]   log (host): ${LOG_FILE}"
  echo "[run_sft]   acompanhar:  tail -f '${LOG_FILE}'   (ou: docker logs -f ${CNAME})"
  echo "[run_sft]   só a loss:   tail -f '${LOG_FILE}' | grep -Ei 'loss|epoch'"
  echo "[run_sft]   status:      docker ps --filter name=${CNAME}"
  echo "[run_sft]   parar:       docker stop ${CNAME}"
  echo "[run_sft]   limpar:      docker rm ${CNAME}   (depois de terminar)"
  exit 0
fi

# ── Primeiro plano: tee no host (stdout ao vivo + arquivo) ───────────────────
echo "[run_sft]   acompanhar a loss: tail -f '${LOG_FILE}' | grep -Ei 'loss|epoch'"
set +e
docker run --rm "${COMMON_ARGS[@]}" "${IMAGE}" \
  accelerate launch --config_file "${LAUNCH_CFG}" \
    --main_process_port "${MAIN_PORT}" \
    --num_processes "${GPUS_PER_NODE}" \
    "${TRAIN_SCRIPT}" "$@" 2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

echo "[run_sft] fim (exit=${status}) — log salvo em: ${LOG_FILE}"
exit "${status}"
