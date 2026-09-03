# Runbook — reproduzir e auditar o caminho até o instruct final<br>End-to-end runbook to reproduce and audit the path to the final instruct

Este documento amarra, em ordem, **todo o caminho** do base até o `manaca-1b-instruct-safe`
(o modelo final), com os comandos exatos e os ponteiros para o código, os dados e
os logs. Só o que funcionou e valeu; o que não funcionou entra como **resultado
negativo documentado** (o DPO), não como ruído.

This ties together, in order, the whole path from the base to the final
`manaca-1b-instruct-safe`, with exact commands and pointers to code, data, and logs.

## O caminho | The path

```
manaca-1b-base  →  SFT v1  →  SFT v2  →  [DPO on-policy: resultado NEGATIVO]  →  safety-SFT  →  safe4 (final)
                   (full FT)  (full FT)   (nao mudou a geracao)                  (AnswerCarefully)   (escolhido)
```

## 0. Ambiente | Environment

```bash
cp .env.example .env          # CKPT_DIR, DATA_DIR, HF_TOKEN, GPUS_PER_NODE=2
make build-posttrain          # imagem GPU de SFT/DPO (trl/accelerate/deepspeed)
./scripts/eval/build_lmeval_image.sh   # imagem de avaliacao (lm-eval + judge)
```

Pré-req.: Docker + NVIDIA Container Toolkit; validado em 2× GPU de 24 GB.

## 1. Corpus de SFT (v1 e v2)

Composições distintas, documentadas em [`docs/data/`](docs/data/) (fontes, splits,
licenças, contagens reais nos `manifest_*.json`, amostras em `sample_*.jsonl`).

```bash
make sft-data       # v1: alpaca + oasst + manaca-jaster(10)   -> data/sft
make sft-data-v2    # v2: alpaca+aya+oasst+traducao+resumo + jaster(12) -> data/sft_v2
```

Código de extração/tratamento: [`sft/prepare_data.py`](sft/prepare_data.py) e
[`sft/jaster/build_jaster.py`](sft/jaster/build_jaster.py).

## 2. Treino SFT (v1 e v2)

Full FT sobre o base, ZeRO-3, grad_accum 128, LR 1e-5, 2 épocas. Guia e tabela em
[`sft/README.md`](sft/README.md); logs reais das corridas vencedoras em
[`docs/training/`](docs/training/).

```bash
DETACH=1 make sft ARGS="--model_name_or_path menezesbruno/manaca-1b-base \
  --data_files data/sft/manaca_sft.jsonl --full_finetuning \
  --gradient_accumulation_steps 128 \
  --output_dir /workspace/checkpoints/manaca-1b-instruct-full"     # v1
DETACH=1 make sft-v2                                               # v2
```

## 3. Alinhamento de segurança

Registro completo em [`docs/evaluation/safety-alignment-pt.md`](docs/evaluation/safety-alignment-pt.md).

### 3a. DPO on-policy — resultado NEGATIVO (documentado)

Fiel ao `llm-jp-dpo` (β=0.1). A curva de recompensa foi perfeita (margem 1.0) mas a
geração livre **não mudou** — o adapter obedecia até os próprios prompts de treino
(diagnóstico [`dpo/diag_dpo.py`](dpo/diag_dpo.py)). Por isso **não** é o método
final. Reprodução em [`dpo/README-onpolicy.md`](dpo/README-onpolicy.md) (`make dpo`).

### 3b. Safety-SFT — método adotado (produz o `safe`)

Estilo AnswerCarefully: recusa como alvo de cross-entropy + ajudar o benigno + dados
gerais. O modelo final (`safe4`) usou a receita **`K_REFUSE=4`, benign-help 50
(disjunto da avaliação), `N_GENERAL=200`**, LoRA r=32, 3 épocas → **411 exemplos**
(~161 recusas + 50 ajudas + 200 gerais).

```bash
K_REFUSE=4 N_GENERAL=200 ./scripts/run_gen_safety_sft.sh   # -> sft_safety/safety_sft.jsonl
make sft-safety                                            # -> manaca-1b-instruct-v2-safe(-merged)
```

Guia: [`sft/README-safety.md`](sft/README-safety.md). Sementes versionadas:
[`dpo/dpo_seeds/safety_prompts.jsonl`](dpo/dpo_seeds/safety_prompts.jsonl) (pedidos
nocivos, só perguntas) e [`sft/safety_seeds/benign_help.jsonl`](sft/safety_seeds/benign_help.jsonl).

## 4. Avaliação | Evaluation

Comparativo (só instructs) em [`docs/evaluation/instruct-eval-pt.md`](docs/evaluation/instruct-eval-pt.md);
harness e evidência crua (respostas + notas do juiz) em [`bench/`](bench/).

```bash
# MT-Bench-PT: gerar -> julgar (juiz LLM) -> reportar
MODEL=<merged> LABEL=<rotulo> ./scripts/eval/run_mtbench_pt.sh
python3 bench/mtbench_pt/judge.py --answers ... --out ...   # precisa ANTHROPIC_API_KEY
python3 bench/mtbench_pt/report.py ... --out ...
# IFEval-PT: gerar (via run_mtbench_pt com QUESTIONS=) -> pontuar
python3 bench/ifeval_pt/score.py <answers...> --prompts bench/ifeval_pt/prompts.jsonl
# Bateria PT-BR: lm-eval -> tabela acc -> f1-macro
./scripts/eval/run_lm_eval_pt.sh ; python3 scripts/eval/merge_ptbench.py ... ; python3 scripts/eval/f1_ptbench.py ...
# Seguranca: sondas offline (recusa held-out ALTA / over-refusal BAIXA)
MODEL=<merged> LABEL=<rotulo> ./scripts/eval/run_overrefusal_pt.sh --prompts bench/safety_pt/heldout.jsonl --out_dir bench/safety_pt/answers
MODEL=<merged> LABEL=<rotulo> ./scripts/eval/run_overrefusal_pt.sh
```

## 5. Usar o modelo final | Use the final model

Template Alpaca-PT (o mesmo do treino). Chat interativo:

```bash
MODEL=<caminho do safe-merged> ./scripts/run_chat.sh
MODEL=<...> ./scripts/run_chat.sh --prompt "Explique o que é fotossíntese."
```

Código: [`scripts/chat.py`](scripts/chat.py). Prompt cru:
`Abaixo está uma instrução...\n\n### Instrução:\n<pergunta>\n\n### Resposta:\n`.

## 6. Obter os pesos | Get the weights

Os pesos completos (dezenas de GB) **não** ficam no git; vão para o Hugging Face
(base: [`menezesbruno/manaca-1b-base`](https://huggingface.co/menezesbruno/manaca-1b-base);
o instruct/safe será publicado no HF). Este repositório contém código, dados
(cards + amostras + manifests) e logs para **reproduzir** os pesos, não os pesos.

## 7. Proveniência e limites de reprodução | Provenance & caveats

Para uma auditoria honesta, os pontos que afetam a reprodução exata:

- **Proveniência amarrada a commits do pipeline base** (`manaca-1b`): as corridas de
  SFT registram o commit exato (v1 `03a0538`, v2 `37491b7`) em
  [`docs/training/`](docs/training/). O código aqui é a **cópia fiel** desse
  pipeline; pequenas melhorias posteriores (ex.: `use_reentrant=False` no
  checkpointing para o DDP do safety-SFT) não afetam v1/v2 (que rodaram em ZeRO-3).
- **Fontes HF sem revisão pinada**: `prepare_data.py`/`build_jaster.py` baixam as
  versões correntes dos datasets; o **manacá-jaster** tem `sha256` por tarefa nos
  manifests (conteúdo pinado), mas as demais fontes podem variar com o tempo.
- **Juiz LLM não é determinístico**: as notas do MT-Bench-PT dependem do modelo juiz
  (claude-opus-5) e podem variar levemente entre execuções e versões do juiz.
- **Sondas pequenas** (segurança 16, over-refusal 34): tratar diferenças de 1–2
  recusas como ruído.
- **Licença dos dados**: a mistura de SFT inclui fontes não-comerciais (Alpaca-PT,
  XLSum); ver a nota em [`docs/data/`](docs/data/).

---

*Manacá-1B-Instruct — LNCC (Instituto de IA) × NII/LLM-jp.*
