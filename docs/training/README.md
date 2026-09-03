# Logs de treino — SFT v1/v2 e alinhamento (safety-SFT) | Training logs

Esta pasta guarda os **logs das corridas que produziram os modelos** — o SFT
(instruct v1 e v2) e o alinhamento de segurança (a fronteira safe/safe2/safe3/safe4,
sendo o **safe4 o modelo final**). Cada log traz a curva de loss (`epoch`,
`learning_rate`, `grad_norm`) e a proveniência amarra a corrida a um commit exato
do pipeline. Só corridas bem-sucedidas — as que falharam ficam listadas ao final,
por transparência, mas fora do registro.

This folder holds the **logs of the runs that produced the models** — the SFT
(instruct v1 and v2) and the safety alignment (the safe/safe2/safe3/safe4 frontier,
with **safe4 as the final model**). Only successful runs are versioned; failed
attempts are listed at the end for honesty but kept out of the record.

## 1. SFT — instruct v1 e v2

| Arquivo | Corrida | Resultado |
|---|---|---|
| `sft_v1_20260831_052916.log` (+ proveniência) | `20260831_052916` (git `03a0538`) | `manaca-1b-instruct-full` · full FT, ZeRO-3, grad_accum 128, 2 ép / 670 passos, loss 2.62→1.64 |
| `sft_v2_20260901_015913.log` (+ proveniência) | `20260901_015913` (git `37491b7`) | `manaca-1b-instruct-v2-full` · full FT, ZeRO-3, grad_accum 128, 2 ép / 1104 passos, loss 2.77→1.59 |

## 2. Safety-SFT — a fronteira segurança × prestatividade

LoRA r=32 sobre o `instruct-v2-full`, DDP. A fronteira é controlada pela razão
recusa : ajuda nos dados; o **safe4 foi o ponto escolhido** (melhor equilíbrio). As
métricas (segurança held-out / over-refusal) estão em
[`../evaluation/safety-alignment-pt.md`](../evaluation/safety-alignment-pt.md).

| Arquivo | Corrida | Receita (recusa/benign/geral, épocas) | Loss | Seg. held-out ↑ | Over-refusal ↓ |
|---|---|---|---|:--:|:--:|
| `sft_safe_20260903_093036.log` | `093036` (git `df7ab85`) | K=2 / 34 (contaminado) / 200 · 3 ép | 1.80→0.78 | 81% | 11.8% |
| `sft_safe2_20260903_094128.log` | `094128` (git `df7ab85`) | K=2 / 34 / 600 · 2 ép | 1.83→1.10 | 69% | 17.6% |
| `sft_safe3_20260903_095609.log` | `095609` (git `3cc30dc`) | K=2 / 50 (disjunto) / 300 · 3 ép | 1.66→0.81 | 56% | 2.9% |
| **`sft_safe4_20260903_100758.log`** ⭐ | `100758` (git `3cc30dc`) | **K=4 (161) / 50 (disjunto) / 200 · 3 ép** | 1.79→0.51 | **75%** | **8.8%** |

⭐ `safe4` = `manaca-1b-instruct-v2-safe4` → o **instruct final (safe)**. Os demais
são os outros pontos da fronteira, versionados para auditar a varredura completa.

## 3. Tentativas descartadas (não versionadas, por transparência)

Falharam na largada (zero treino) e **não** entram no registro:

- SFT v1: `20260831_050317`, `051551`, `052348` — `ChildFailedError` (falha de processo).
- SFT v2: `20260901_013034` — `EADDRINUSE` (porta 29500 ocupada por outro run).
- Safety-SFT: `20260903_034535` — `CheckpointError` do gradient checkpointing
  reentrante sob DDP (corrigido depois com `use_reentrant=False`).

Only successful runs are versioned; the failed attempts above are listed for process
honesty but intentionally kept out of the record.
