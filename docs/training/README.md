# Logs de treino — SFT v1 e v2 | Training logs — SFT v1 and v2

Esta pasta guarda os **logs das corridas de SFT que produziram os modelos
publicados** (instruct v1 e v2): as curvas de loss e a proveniência de cada run
(commit, imagem Docker, hiperparâmetros, GPUs). Só as corridas que valeram — sem
as tentativas descartadas — para o registro ficar enxuto e auditável.

This folder holds the **logs of the SFT runs that produced the published models**
(instruct v1 and v2): the loss curves and the provenance of each run (commit,
Docker image, hyperparameters, GPUs). Only the runs that mattered — no discarded
attempts — to keep the record lean and auditable.

## Arquivos | Files

| Arquivo | Corrida | Resultado |
|---|---|---|
| `sft_v1_20260831_052916.log` + `provenance_sft_v1_...txt` | `20260831_052916` (git `03a0538`) | `manaca-1b-instruct-full` · full FT, ZeRO-3, grad_accum 128, 2 ép / 670 passos, loss 2.62→1.64 |
| `sft_v2_20260901_015913.log` + `provenance_sft_v2_...txt` | `20260901_015913` (git `37491b7`) | `manaca-1b-instruct-v2-full` · full FT, ZeRO-3, grad_accum 128, 2 ép / 1104 passos, loss 2.77→1.59 |

Cada linha de `loss` traz `epoch`, `learning_rate` e `grad_norm`, permitindo
reconstruir a curva de treino. A proveniência amarra cada corrida a um commit
exato do pipeline.

### Tentativas descartadas (não versionadas, por transparência)

Estes runs falharam na largada (zero treino) e **não** entram no registro, para
mantê-lo enxuto — ficam listados aqui só para honestidade do processo:

- v1: `20260831_050317`, `051551`, `052348` — `ChildFailedError` (falha de processo).
- v2: `20260901_013034` — `EADDRINUSE` (porta 29500 ocupada por outro run).

Only the successful runs are versioned; the early failed attempts above are listed
for process honesty but intentionally kept out of the record.
