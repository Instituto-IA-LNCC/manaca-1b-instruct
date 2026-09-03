# =============================================================================
# Manacá-1B-Instruct — pós-treino (SFT) reprodutível. Atalhos do que VALEU.
# -----------------------------------------------------------------------------
# Só os alvos do instruct v1 e v2 (o que funcionou). A base (pré-treino/aval.)
# está em github.com/Instituto-IA-LNCC/manaca-1b-base.
#
#   cp .env.example .env    # ajuste CKPT_DIR, DATA_DIR, HF_TOKEN, GPUS_PER_NODE=2
#   make build-posttrain    # imagem GPU (trl/accelerate/deepspeed)
#   make sft-data           # dados v1  |  make sft-data-v2  -> dados v2
#   make sft ARGS="..."     # SFT v1    |  make sft-v2       -> SFT v2
# =============================================================================
SHELL := /bin/bash
DC := docker compose
.PHONY: help env build-posttrain sft-data sft-data-v2 sft sft-v2
.RECIPEPREFIX := >

help:  ## Lista os alvos
> @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

env:  ## Cria .env a partir do exemplo, se faltar
> @[ -f .env ] || { cp .env.example .env && echo "[env] .env criado a partir de .env.example — ajuste os caminhos"; }

build-posttrain: env  ## Constrói a imagem GPU de SFT (Fase 3)
> $(DC) build posttrain

sft-data:  ## Dados de SFT v1 (manaca-jaster + alpaca + oasst) -> data/sft
> ./scripts/docker/run_posttrain.sh bash -lc 'python sft/jaster/build_jaster.py --tasks default --out data/sft/jaster --shuffle && python sft/prepare_data.py --sources alpaca,oasst --out data/sft --shuffle --dedup --extra data/sft/jaster/manaca_jaster.jsonl'

sft-data-v2:  ## Dados de SFT v2 (OASST corrigido + aya + traducao + sumarizacao + jaster completo; filtro de longos) -> data/sft_v2
> ./scripts/docker/run_posttrain.sh bash -lc 'python sft/jaster/build_jaster.py --tasks all --out data/sft_v2/jaster --shuffle --max_chars 5000 --max_per_task 8000 && python sft/prepare_data.py --sources alpaca,aya,oasst,translation,summarization --out data/sft_v2 --shuffle --dedup --max_chars 5000 --extra data/sft_v2/jaster/manaca_jaster.jsonl'

sft:  ## SFT (Fase 3a): make sft ARGS="..."  (DETACH=1 para segundo plano)
> ./scripts/docker/run_sft.sh sft $(ARGS)

sft-v2:  ## SFT v2 (full FT) -> checkpoints/manaca-1b-instruct-v2-full (NAO sobrescreve a v1). DETACH=1 p/ 2o plano
> ./scripts/docker/run_sft.sh sft --model_name_or_path menezesbruno/manaca-1b-base --data_files data/sft_v2/manaca_sft.jsonl --full_finetuning --gradient_accumulation_steps 128 --output_dir /workspace/checkpoints/manaca-1b-instruct-v2-full $(ARGS)
