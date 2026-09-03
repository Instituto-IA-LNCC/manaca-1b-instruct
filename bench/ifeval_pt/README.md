# IFEval-PT — seguir instrução (verificável por regra, sem juiz)

Avaliação de **instruction following** no espírito do IFEval (Zhou et al. 2023),
adaptada ao PT-BR. Mede se o modelo **obedece a restrições verificáveis por código**
(contagem de palavras/frases/parágrafos, incluir/evitar palavra, maiúsculas, JSON
válido, terminar com frase exata, etc.). Não usa juiz, então roda de graça e é
100% reprodutível. É a categoria "IF" da bateria do LLM-jp.

## Conjunto

`prompts.jsonl` — 36 prompts, 50 instruções verificáveis. Cada linha:
`{id, category, turns:[prompt], reference:null, instructions:[{type, kwargs}]}`.
Os tipos de instrução e seus checadores estão em `checkers.py`.

## Métricas (padrão IFEval)

- **prompt-strict**: % de prompts em que TODAS as instruções passam (resposta crua).
- **prompt-loose**: idem, com normalização (remove preâmbulo/1ª e última linha, markdown, cercas de código, aspas) e passa se qualquer variante satisfizer.
- **instr-strict / instr-loose**: idem, mas por instrução individual.

## Fluxo (3 passos)

### 1. Gerar (reusa o runner do MT-Bench, apontando para o IFEval)

Locais (Manacá) e do HF (baselines), cada um com seu `chat_template`:

```bash
cd /prj/prjgvdc/brunolsm/manaca-1b
Q=bench/ifeval_pt/prompts.jsonl; A=bench/ifeval_pt/answers

QUESTIONS=$Q ANS_DIR=$A MODEL=/data/brunolsm/manaca-checkpoints/manaca-1b-instruct-full \
  LABEL=manaca-instruct-v1 ./scripts/eval/run_mtbench_pt.sh
QUESTIONS=$Q ANS_DIR=$A MODEL=/data/brunolsm/manaca-checkpoints/manaca-1b-instruct-v2-full \
  LABEL=manaca-instruct-v2 ./scripts/eval/run_mtbench_pt.sh

QUESTIONS=$Q ANS_DIR=$A HF_CACHE=$HOME/hf_cache_mtbench \
  HFID=TucanoBR/Tucano-1b1-Instruct LABEL=tucano-1b1-instruct ./scripts/eval/run_mtbench_pt.sh
QUESTIONS=$Q ANS_DIR=$A HF_CACHE=$HOME/hf_cache_mtbench \
  HFID=TucanoBR/Tucano-2b4-Instruct LABEL=tucano-2b4-instruct ./scripts/eval/run_mtbench_pt.sh
QUESTIONS=$Q ANS_DIR=$A HF_CACHE=$HOME/hf_cache_mtbench \
  HFID=nicholasKluge/TeenyTinyLlama-460m-Chat LABEL=ttl-460m-chat ./scripts/eval/run_mtbench_pt.sh
```

### 2. Pontuar (sem juiz, só regra)

```bash
python3 bench/ifeval_pt/score.py \
  bench/ifeval_pt/answers/manaca-instruct-v1.jsonl \
  bench/ifeval_pt/answers/manaca-instruct-v2.jsonl \
  bench/ifeval_pt/answers/tucano-1b1-instruct.jsonl \
  bench/ifeval_pt/answers/tucano-2b4-instruct.jsonl \
  bench/ifeval_pt/answers/ttl-460m-chat.jsonl \
  --out bench/ifeval_pt/ifeval-pt
```

Gera `ifeval-pt.md` (tabela strict/loose por modelo + instr-loose por tipo) e `.json`.

### 3. Commitar (transparência)

`prompts.jsonl`, `answers/*.jsonl`, `checkers.py`, `score.py` e `ifeval-pt.md/.json`
vão para o git. Aqui **não há saída nociva** (nenhum prompt de segurança), então
não precisa redigir nada.
