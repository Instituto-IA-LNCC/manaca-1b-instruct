# DPO on-policy — fiel ao LLM-jp, adaptado à nossa realidade

Corrige o 1º DPO (que usou pares off-policy do GigaVerbo e não melhorou / sobre-otimizou).
Aqui os pares de preferência são gerados **pelo próprio instruct** (on-policy) e
focam nos gaps que os benchmarks mediram: **segurança** (o modelo obedece a pedidos
nocivos) e **seguir instrução**. Método fiel ao LLM-jp (dados por juiz + linha de
segurança tipo AnswerCarefully); adaptações: on-policy, PT-BR, poucas épocas com
parada pelo sinal, e decisão pela avaliação (MT-Bench-PT / IFEval-PT).

## 1. Gerar os pares (na HPC, GPU)

```bash
cd /prj/prjgvdc/brunolsm/manaca-1b

# (a) SEGURANCA — regra, offline. Maior retorno. (rejected = obediencia do modelo)
DETACH=1 MODE=safety      ./scripts/run_gen_dpo_pairs.sh

# (b) INSTRUCAO — regra (checkers do IFEval), offline
DETACH=1 MODE=instruction ./scripts/run_gen_dpo_pairs.sh

# (c) QUALIDADE — juiz LLM (precisa de chave)
DETACH=1 MODE=quality ANTHROPIC_API_KEY=sk-ant-... ./scripts/run_gen_dpo_pairs.sh
```

Saída em `dpo_onpolicy/<mode>.jsonl` (esquema `{instruction, input, chosen, rejected}`).
**Não versionado** (o `rejected` de segurança é conteúdo nocivo; `.gitignore` cobre `dpo_onpolicy/`).

Junte os fluxos que quiser num arquivo só:
```bash
cat dpo_onpolicy/safety.jsonl dpo_onpolicy/instruction.jsonl > dpo_onpolicy/mix.jsonl
```

## 2. Treinar o DPO (poucas épocas + parada pelo sinal)

Fiel ao `llm-jp-dpo` (LoRA, LR 5e-7, ref = SFT v2), mas **1-2 épocas** e salvando por
época, para escolher o checkpoint antes da sobre-otimização (o erro do 1º DPO):

```bash
make dpo ARGS="--model_name_or_path /workspace/checkpoints/manaca-1b-instruct-v2-full \
  --data_files dpo_onpolicy/mix.jsonl \
  --output_dir /workspace/checkpoints/manaca-1b-instruct-v2-dpo2 \
  --beta 0.1 --learning_rate 5e-7 --num_train_epochs 2 \
  --save_strategy epoch --save_total_limit 5 --lora_r 64"
```

O alvo `make dpo` usa `configs/accelerate_config_ddp.yaml` por default (DDP, não
ZeRO-3): para LoRA de 1.7B o modelo cabe folgado em 1 GPU, e o ZeRO-3, ao
particionar os pesos, quebra o gradient checkpointing do DPO (o adapter liga/
desliga entre política e referência) com `CheckpointError` de metadados. Para
sobrepor: `ACCELERATE_CONFIG=configs/accelerate_config_zero3.yaml make dpo ...`
(necessário apenas para full FT).

No log, observe `rewards/chosen` e `rewards/margins`: **pare/escolha o checkpoint
antes de `rewards/chosen` virar negativo** (sinal de sobre-otimização). Se derivar
rápido, suba `--beta` para 0.2-0.3 (mantém o modelo mais perto do SFT).

## 3. Avaliar (só fica com o DPO se melhorar)

Comparar o DPO contra o **instruct-v2** com a nossa avaliação:
- **MT-Bench-PT, categoria `seguranca`** deve subir bastante (é o alvo).
- **Over-refusal** (`bench/overrefusal_pt`) não pode subir — o risco do mix ser
  96% "recuse" é o modelo virar recusador demais e negar pedidos legítimos.
- **MT-Bench-PT geral + IFEval-PT** não podem regredir.
- CALAME/LAMBADA não podem cair (base intacta).

```bash
# over-refusal: ANTES (SFT) e DEPOIS (DPO). 'sensivel_legitimo' nao pode subir.
LABEL=manaca-instruct-v2 ./scripts/eval/run_overrefusal_pt.sh
MODEL=/data/brunolsm/manaca-checkpoints/manaca-1b-instruct-v2-dpo2-merged \
  LABEL=manaca-instruct-v2-dpo2 ./scripts/eval/run_overrefusal_pt.sh
```

```bash
# gera respostas do DPO e julga (mesma infra do bench)
MODEL=/data/brunolsm/manaca-checkpoints/manaca-1b-instruct-v2-dpo2-merged \
  LABEL=manaca-instruct-v2-dpo2 ./scripts/eval/run_mtbench_pt.sh
python3 bench/mtbench_pt/judge.py --answers bench/mtbench_pt/answers/manaca-instruct-v2-dpo2.jsonl \
  --out bench/mtbench_pt/judged/manaca-instruct-v2-dpo2.jsonl
python3 bench/mtbench_pt/report.py \
  bench/mtbench_pt/judged/manaca-instruct-v2.jsonl \
  bench/mtbench_pt/judged/manaca-instruct-v2-dpo2.jsonl --out bench/mtbench_pt/mtbench-pt-dpo
```

Se `seguranca` sobe e o resto se mantém, o DPO valeu e vira o `instruct-v2-dpo` oficial.
Senão, o instruct-v2 (SFT) continua sendo o melhor, e a evidência fica registrada.
