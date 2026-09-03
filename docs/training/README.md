# Logs de treino — SFT v1 e v2 | Training logs — SFT v1 and v2

Esta pasta guarda os **logs das corridas de SFT que produziram os modelos
publicados** (instruct v1 e v2): as curvas de loss e a proveniência de cada run
(commit, imagem Docker, hiperparâmetros, GPUs). Só as corridas que valeram — sem
as tentativas descartadas — para o registro ficar enxuto e auditável.

This folder holds the **logs of the SFT runs that produced the published models**
(instruct v1 and v2): the loss curves and the provenance of each run (commit,
Docker image, hyperparameters, GPUs). Only the runs that mattered — no discarded
attempts — to keep the record lean and auditable.

## Arquivos esperados | Expected files

| Arquivo | Conteúdo |
|---|---|
| `sft_v1_<timestamp>.log` | Corrida de SFT que gerou `manaca-1b-instruct-full` (v1) |
| `provenance_sft_v1_<timestamp>.txt` | Proveniência do run v1 (commit, args, imagem) |
| `sft_v2_<timestamp>.log` | Corrida de SFT que gerou `manaca-1b-instruct-v2-full` (v2) |
| `provenance_sft_v2_<timestamp>.txt` | Proveniência do run v2 |

Os logs originais ficam na HPC em `${CKPT_DIR}/sft-logs/`. Cada linha de `loss`
traz `epoch`, `learning_rate` e `grad_norm`, permitindo reconstruir a curva de
treino. A proveniência amarra a corrida a um commit exato deste repositório.
