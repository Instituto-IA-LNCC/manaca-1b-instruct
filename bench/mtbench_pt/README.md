# MT-Bench-PT — avaliação de instruct por LLM-as-a-Judge

Avaliação **generativa** dos modelos instruct do Manacá, no espírito do LLM-jp
(ver `references/phase4-evaluation/llm-jp-eval-pipeline-analise.md`, §15). Enquanto
ARC/HellaSwag/LAMBADA/CALAME medem a **capacidade de base**, este benchmark mede o
que o SFT/DPO de fato mudam: **seguir instrução, qualidade e utilidade da resposta,
e recusa de pedidos perigosos**.

Método fiel ao LLM-jp:
- geração **com o template de instrução** do próprio modelo (Alpaca-PT do SFT), não em modo base;
- **single-answer grading**: um LLM forte julga cada resposta de forma **independente** (sem comparar A com B), evitando viés de posição;
- **rubrica estrita de 1 a 10** (estilo Heron-Bench); tarefas objetivas recebem a `reference` como gabarito;
- categoria **seguranca** com rubrica invertida (recusar bem = nota alta).

## Conjunto de perguntas

`questions.jsonl` — 54 perguntas, 6 por categoria, em 9 categorias:
`escrita`, `interpretacao`, `raciocinio`, `matematica`, `codigo`, `extracao`,
`ciencias`, `humanas`, `seguranca`. Cada linha: `{id, category, turns, reference}`.
(É single-turn nesta versão; `turns` é lista para permitir multi-turn no futuro.)

## Fluxo (3 passos)

### 1. Gerar as respostas de cada modelo (na HPC, com GPU)

```bash
cd /prj/prjgvdc/brunolsm/manaca-1b

MODEL=/data/brunolsm/manaca-checkpoints/manaca-1b-instruct-full \
LABEL=manaca-instruct-v1 ./scripts/eval/run_mtbench_pt.sh

MODEL=/data/brunolsm/manaca-checkpoints/manaca-1b-instruct-v2-full \
LABEL=manaca-instruct-v2 ./scripts/eval/run_mtbench_pt.sh
```

Saída: `bench/mtbench_pt/answers/<label>.jsonl`. A geração é offline (não baixa nada)
e roda com `--user`, então os arquivos já nascem com o seu dono.

### 1b. Baselines instruct (do Hugging Face)

Para comparar **instruct vs instruct** de forma justa, cada modelo é interrogado com
o **seu próprio `chat_template`** (`--prompt_style auto`, o default). O Manacá, que
não tem `chat_template`, usa o Alpaca-PT do SFT; um Tucano-Instruct usa o template dele.

Use `HFID=` para baixar do HF (o wrapper monta o cache e liga a rede):

```bash
HFID=TucanoBR/Tucano-1b1-Instruct LABEL=tucano-1b1-instruct ./scripts/eval/run_mtbench_pt.sh
HFID=TucanoBR/Tucano-2b4-Instruct LABEL=tucano-2b4-instruct ./scripts/eval/run_mtbench_pt.sh
HFID=nicholasKluge/TeenyTinyLlama-460m-Chat LABEL=ttl-460m-chat ./scripts/eval/run_mtbench_pt.sh
```

Confirme os IDs que existem de fato (na HPC, com acesso ao HF):
```bash
python3 - <<'PY'
from huggingface_hub import list_models
for a in ["TucanoBR", "nicholasKluge", "maritaca-ai"]:
    print("==", a)
    for m in list_models(author=a):
        if any(t in m.id.lower() for t in ("instruct", "chat", "-it")):
            print("  ", m.id)
PY
```

Nota: **Sabiá-7B (`maritaca-ai/sabia-7b`) é só base**; o chat do Sabiá é fechado (API),
então ele não entra na comparação de instruct. GlorIA e mGPT também são só base.

### 2. Julgar (precisa de um LLM juiz)

O juiz tem dois provedores. Ele detecta a Anthropic automaticamente se
`ANTHROPIC_API_KEY` estiver definida.

**Anthropic (recomendado, API nativa de Mensagens):**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export JUDGE_MODEL=claude-opus-5     # opcional; default ja e claude-opus-5

# valide o juiz primeiro (esperado ~ vazia:1 / sem_sentido:1-2 / perfeita:9-10)
python3 bench/mtbench_pt/judge.py --sanity

# julgue cada modelo
python3 bench/mtbench_pt/judge.py \
  --answers bench/mtbench_pt/answers/manaca-instruct-v1.jsonl \
  --out     bench/mtbench_pt/judged/manaca-instruct-v1.jsonl
python3 bench/mtbench_pt/judge.py \
  --answers bench/mtbench_pt/answers/manaca-instruct-v2.jsonl \
  --out     bench/mtbench_pt/judged/manaca-instruct-v2.jsonl
```

**OpenAI-compatível (OpenAI, Azure, ou um vLLM local no cluster):**
```bash
export JUDGE_PROVIDER=openai
export JUDGE_BASE_URL=https://api.openai.com/v1   # ou http://localhost:8000/v1 (vLLM local)
export JUDGE_MODEL=gpt-4o
export JUDGE_API_KEY=sk-...                        # 'x' para vLLM local sem auth
python3 bench/mtbench_pt/judge.py --sanity
# ... mesmos comandos de --answers/--out acima
```

O juiz roda onde houver acesso à API (a HPC, se tiver saída para a internet, ou a
sua máquina após copiar os `answers/`). Só usa a biblioteca padrão (nenhum `pip install`).

> **Escolha do juiz (viés).** GPT-4o/Claude como juiz tendem a favorecer respostas
> verbosas e em inglês (ver §18 da análise). Documente sempre qual juiz foi usado.
> O ideal, quando possível, é rodar com dois juízes diferentes e reportar ambos.

### 3. Agregar e comparar v1 vs v2

```bash
python3 bench/mtbench_pt/report.py \
  bench/mtbench_pt/judged/manaca-instruct-v1.jsonl \
  bench/mtbench_pt/judged/manaca-instruct-v2.jsonl \
  --out bench/mtbench_pt/mtbench-pt
```

Gera `mtbench-pt.md` (tabela: nota média ± erro padrão por categoria + GERAL, e o
delta v2 − v1) e `mtbench-pt.json`.

## O que fica versionado no git (transparência)

`questions.jsonl`, os `answers/*.jsonl` (respostas geradas), os `judged/*.jsonl`
(notas + justificativa do juiz + saída bruta) e a tabela `mtbench-pt.md/.json`.
Assim qualquer pessoa reproduz e audita o julgamento.
