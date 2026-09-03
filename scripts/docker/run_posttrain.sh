#!/usr/bin/env bash
# =============================================================================
# Manacá — roda um comando dentro da imagem manaca-posttrain (SFT/DPO utils)
# =============================================================================
# Para preparo de dados e utilitários da Fase 3 (build_jaster, prepare_data,
# merge_lora...). Monta o repo, DATA_DIR, HF_CACHE_DIR e CKPT_DIR, e o DNS REAL
# do host (via resolvectl) para resolver o HuggingFace. NÃO reserva GPU.
#
# Uso:
#   ./scripts/docker/run_posttrain.sh python sft/prepare_data.py --sources bode,oasst ...
#   ./scripts/docker/run_posttrain.sh bash -lc 'python a.py && python b.py'
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Carrega .env como DEFAULTS (ambiente/CLI têm precedência).
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
mkdir -p "${DATA_DIR}" "${HF_CACHE_DIR}" "${CKPT_DIR}"

if [ "$#" -eq 0 ]; then
    echo "[run_posttrain] uso: $0 <comando...>  (ex.: python sft/prepare_data.py ...)"; exit 2
fi

# DNS real do host (mesmo mecanismo do run_sft.sh). Override: MANACA_DNS="ns1 ns2".
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

exec docker run --rm --network host --ipc host \
  --shm-size=16g \
  "${RESOLV_MOUNT[@]}" \
  -e WORK_DIR=/workspace/manaca-corpus \
  -e HF_HOME=/workspace/hf-cache \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e PIP_CONFIG_FILE=/dev/null \
  -v "${REPO_ROOT}":/workspace/manaca \
  -v "${DATA_DIR}":/workspace/manaca-corpus \
  -v "${HF_CACHE_DIR}":/workspace/hf-cache \
  -v "${CKPT_DIR}":/workspace/checkpoints \
  -w /workspace/manaca \
  "${IMAGE}" \
  "$@"
