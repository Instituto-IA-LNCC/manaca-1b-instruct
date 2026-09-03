# Safety-SFT — segurança pelo método que funciona (estilo AnswerCarefully)

> **Resultado e modelo escolhido:** ver `docs/evaluation/safety-alignment-pt.md`.
> O ponto escolhido da fronteira foi o **safe4** (75% de segurança held-out,
> 8.8% de over-refusal), e a receita dele é o **default** deste fluxo
> (`K_REFUSE=4`, benign-help disjunto, `N_GENERAL=200`, 3 épocas).

## Por que SFT e não DPO

O DPO on-policy gentil (fiel ao LLM-jp: beta 0.1, LR baixo) **não reverteu** o
prior de obediência do Manacá. O diagnóstico (`dpo/diag_dpo.py`) mostrou que o
adapter do DPO — carregado direto, sem merge — **obedece até os próprios prompts
de treino** (0/4), apesar da margem de treino ter chegado a 1.0. A margem mede a
razão de probabilidade das sequências específicas sob teacher-forcing; ela não
vira o primeiro token da geração livre quando o prior de obediência é forte.

A segurança do LLM-jp **não veio de DPO** — veio do **AnswerCarefully**, um
dataset de **SFT de segurança**. SFT otimiza a recusa como ALVO de cross-entropy,
o que muda a geração de verdade. Este é o pivô: fiel ao que o LLM-jp de fato fez.

## 1. Montar o dataset (na HPC, GPU)

Balanceado para não virar recusador cego: **recusar o nocivo** + **ajudar o
benigno** (respostas úteis geradas pelo próprio SFT, on-policy) + **dados gerais**
(amostra do SFT v2, preserva a utilidade ampla).

```bash
cd $HOME/manaca-1b
DETACH_OK=; K_REFUSE=2 N_GENERAL=200 GPUS=device=0 ./scripts/run_gen_safety_sft.sh
wc -l sft_safety/safety_sft.jsonl
```

Saída `sft_safety/safety_sft.jsonl` (`{instruction, input, output}`). O `output` é
sempre recusa OU resposta útil (nunca conteúdo nocivo), mas o arquivo mistura
dados gerais reamostrados, então fica **fora do git** (`.gitignore`); a receita
(prompts em `dpo/dpo_seeds/safety_prompts.jsonl` + banco de recusas em
`sft/gen_safety_sft.py`) já está versionada.

## 2. Treinar (LoRA sobre o v2, DDP, poucas épocas)

```bash
DETACH=1 make sft-safety
#   -> adapter em manaca-1b-instruct-v2-safe, mesclado em manaca-1b-instruct-v2-safe-merged
#   acompanhar: tail -f /data/manaca/checkpoints/sft-logs/sft_*.log | grep -Ei 'loss|epoch'
```

Usa DDP (não ZeRO-3) e LoRA r=32, 3 épocas, LR 1e-4. Não sobrescreve o
`instruct-v2-full`. Se quiser mesclar em fp32 (evita arredondar deltas do LoRA):

```bash
make sft-safety ARGS="--no_merge"
./scripts/docker/run_posttrain.sh python sft/merge_lora.py \
  --base /workspace/checkpoints/manaca-1b-instruct-v2-full \
  --adapter /workspace/checkpoints/manaca-1b-instruct-v2-safe \
  --out /workspace/checkpoints/manaca-1b-instruct-v2-safe-merged
```

## 3. Avaliar (o mesmo tribunal do DPO)

```bash
M=/data/manaca/checkpoints/manaca-1b-instruct-v2-safe-merged

# (a) SEGURANÇA held-out — recusa deve SUBIR muito (era 0/16 no SFT)
MODEL=$M LABEL=safe ./scripts/eval/run_overrefusal_pt.sh \
  --prompts bench/safety_pt/heldout.jsonl --out_dir bench/safety_pt/answers

# (b) OVER-REFUSAL — recusa em benignos NÃO pode subir (era 0% no SFT)
MODEL=$M LABEL=safe ./scripts/eval/run_overrefusal_pt.sh

# (c) diagnóstico rápido lado a lado
BASE=manaca-1b-instruct-v2-full ADAPTER=manaca-1b-instruct-v2-safe \
  MERGED=manaca-1b-instruct-v2-safe-merged ./scripts/eval/run_diag_dpo.sh
```

Depois, se (a) subiu e (b) ficou estável, rode a utilidade (IFEval-PT offline +
MT-Bench-PT com juiz) para confirmar que não regrediu. Só então o
`instruct-v2-safe` vira o instruct oficial.
