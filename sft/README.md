# SFT — treino do Manacá-1B-Instruct v1 e v2<br>Supervised fine-tuning of Manacá-1B-Instruct v1 and v2

Código de treino que **de fato produziu** os instructs v1 e v2, fiel ao
`llm-jp-sft` com a API moderna do `trl` (`SFTConfig`), no template Alpaca-PT.
Aqui está só o que funcionou e valeu — pronto para auditar e reproduzir.

Training code that **actually produced** instruct v1 and v2, faithful to
`llm-jp-sft` with the modern `trl` API (`SFTConfig`), in the Alpaca-PT template.
Only what worked and mattered — ready to audit and reproduce.

## Arquivos | Files

| Arquivo | Papel |
|---|---|
| `train.py` | Treinador SFT (Alpaca-PT, completion-only, full FT ou LoRA) |
| `prepare_data.py` | Monta a mistura de dados (alpaca/aya/oasst/translation/summarization) |
| `jaster/build_jaster.py` | Constrói o `manaca-jaster` (tarefas PT-BR estilo jaster do LLM-jp) |
| `merge_lora.py` | Mescla um adapter LoRA no base (só no modo LoRA) |
| `../scripts/docker/run_sft.sh` | Lança o `accelerate` dentro do container |
| `../configs/accelerate_config_zero3.yaml` | DeepSpeed ZeRO-3 (usado no full FT) |

## Hiperparâmetros | Hyperparameters

Fiéis ao LLM-jp (full FT LR 1e-5, 2 épocas; LoRA LR 1e-4, 5 épocas), Alpaca-PT,
`max_seq_length` 2048, bf16, gradient checkpointing.

| | v1 | v2 |
|---|---|---|
| Base | `menezesbruno/manaca-1b-base` | idem |
| Dados | alpaca + oasst + manaca-jaster (default) | alpaca + aya + oasst + translation + summarization + jaster completo, filtro de longos |
| Modo | full FT | full FT |
| Saída | `manaca-1b-instruct-full` | `manaca-1b-instruct-v2-full` |

## Reproduzir | Reproduce

Pré-requisitos: Docker + NVIDIA Container Toolkit; `GPUS_PER_NODE=2` no `.env`.

```bash
cp .env.example .env            # ajuste CKPT_DIR, DATA_DIR, HF_TOKEN, GPUS_PER_NODE
make build-posttrain            # imagem GPU (trl/accelerate/deepspeed)

# ---------- Instruct v1 ----------
make sft-data                   # -> data/sft/manaca_sft.jsonl
DETACH=1 make sft ARGS="--model_name_or_path menezesbruno/manaca-1b-base \
  --data_files data/sft/manaca_sft.jsonl --full_finetuning \
  --learning_rate 1e-5 --num_train_epochs 2 \
  --output_dir /workspace/checkpoints/manaca-1b-instruct-full"

# ---------- Instruct v2 ----------
make sft-data-v2                # -> data/sft_v2/manaca_sft.jsonl
DETACH=1 make sft-v2            # full FT, grad_accum 128 -> manaca-1b-instruct-v2-full
```

O `run_sft.sh` grava o log e a proveniência (commit, imagem, hiperparâmetros) em
`${CKPT_DIR}/sft-logs/` a cada corrida. O modo full FT salva o **modelo completo**
direto em `--output_dir` (artefato autossuficiente).

Detalhes da arquitetura do modelo base: [`manaca-1b-base`](https://github.com/Instituto-IA-LNCC/manaca-1b-base).

## Logs | Logs

Os logs de treino bem-sucedidos do v1 e v2 (curvas de loss + proveniência) estão
em [`../docs/training/`](../docs/training/). São exatamente as corridas que
produziram os modelos publicados — sem as tentativas descartadas.

The successful v1/v2 training logs (loss curves + provenance) are in
[`../docs/training/`](../docs/training/) — exactly the runs that produced the
published models, without the discarded attempts.
