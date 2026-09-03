# Corpus de SFT do Manacá-1B-Instruct (v1 e v2)<br>SFT corpus of Manacá-1B-Instruct (v1 and v2)

Ficha dos dados de instrução que treinaram o instruct. **A v1 e a v2 usaram
composições diferentes** — aqui estão as fontes exatas, os splits, o tratamento e
os comandos que reproduzem cada mistura. Só o que de fato foi usado, para auditar
e reproduzir sem ambiguidade.

Data card for the instruction data behind the instruct. **v1 and v2 used different
mixes** — exact sources, splits, processing, and the commands that reproduce each.
Only what was actually used, for unambiguous auditing and reproduction.

O código de extração/tratamento é [`sft/prepare_data.py`](../../sft/prepare_data.py)
(mix de instrução/conversa) e [`sft/jaster/build_jaster.py`](../../sft/jaster/build_jaster.py)
(o **manacá-jaster**, análogo PT-BR do Jaster do LLM-jp). Ambos emitem o esquema
Alpaca `{instruction, input, output}` e gravam um `manifest.json` com as contagens.

## Composição por versão | Per-version composition

| | **v1** (`data/sft`) | **v2** (`data/sft_v2`) |
|---|---|---|
| Fontes de instrução/conversa | alpaca, oasst | alpaca, aya, oasst, translation, summarization |
| manacá-jaster | `--tasks default` (10 tarefas) | `--tasks all` (12 tarefas) |
| Dedup | sim | sim |
| Shuffle (seed 42) | sim | sim |
| `max_chars` (corta prompt longo) | — | 5000 |
| `max_per_task` (jaster) | — | 8000 |
| Saída | `data/sft/manaca_sft.jsonl` | `data/sft_v2/manaca_sft.jsonl` |

As contagens exatas por fonte estão nos manifestos versionados nesta pasta
(`manifest_v1.json`, `manifest_v2.json`, `jaster_manifest_v1.json`,
`jaster_manifest_v2.json`, com `sha256` por tarefa do jaster) e um recorte de
exemplos em `sample_v1.jsonl` / `sample_v2.jsonl`.

### Contagens reais | Actual counts (dedup + `max_chars` aplicados)

| Fonte (no mix) | v1 | v2 |
|---|--:|--:|
| alpaca | 51.747 | 51.747 |
| aya | — | 8.809 |
| oasst | **0** | 315 |
| translation (opus-100) | — | 38.754 |
| summarization (xlsum) | — | 4.218 |
| manacá-jaster | 34.145 | 37.561 |
| **TOTAL** | **85.892** | **141.404** |

Dois pontos honestos de auditoria (visíveis nos manifests):

- **v1: OASST = 0.** O filtro de OASST da v1 exigia o idioma também no nó-pai e
  acabou retornando **zero** exemplos — ou seja, a v1 foi, na prática, alpaca +
  manacá-jaster. Esse foi um dos motivos do v2, que relaxou o filtro (315
  exemplos). Registrado aqui em vez de escondido.
- **v2: macmorpho = 0.** A tarefa `macmorpho` do jaster não carregou/ficou vazia no
  build do v2 (as demais 11 tarefas entraram normalmente). Um dataset fora do ar
  é pulado com aviso, sem derrubar o build.

## Fontes de instrução/conversa (`prepare_data.py`)

| Fonte | HF id | Config | Split | v1 | v2 | Licença (verificar no HF) |
|---|---|---|:--:|:--:|:--:|---|
| alpaca | `dominguesm/alpaca-data-pt-br` | — | train | ✓ | ✓ | CC BY-NC 4.0 (linhagem Alpaca) |
| aya | `CohereForAI/aya_dataset` (pt) | — | train | | ✓ | Apache-2.0 |
| oasst | `OpenAssistant/oasst1` (lang=pt) | — | train | ✓ | ✓ | Apache-2.0 |
| translation | `Helsinki-NLP/opus-100` | en-pt | train[:20000] | | ✓ | aberta (OPUS) |
| summarization | `csebuetnlp/xlsum` | portuguese | train[:10000] | | ✓ | CC BY-NC-SA 4.0 |

Tratamento: alpaca (instruction/input/output direto); aya (filtra português por
`language_code`/`language`); oasst (pares prompter→assistant em pt, melhor resposta
por nó); opus-100 (tradução nos **dois** sentidos, PT↔EN); xlsum (texto→resumo).

## manacá-jaster — tarefas de NLP PT-BR (`build_jaster.py`)

Converte datasets de NLP em PT-BR para instrução curta (como o Jaster faz com
JNLI/JSQuAD etc. no japonês).

| Tarefa | HF id | Config | Split | default (v1) | all (v2) |
|---|---|---|:--:|:--:|:--:|
| assin2_sts | `assin2` | — | train | ✓ | ✓ |
| assin2_nli | `assin2` | — | train | ✓ | ✓ |
| sick_br_nli | `eduagarcia/sick-br` | — | train | ✓ | ✓ |
| sick_br_sts | `eduagarcia/sick-br` | — | train | | ✓ |
| faquad | `eraldoluis/faquad` | — | train | ✓ | ✓ |
| enem | `eduagarcia/enem_challenge` | — | train | ✓ | ✓ |
| harem | `harem` | selective | train | ✓ | ✓ |
| lener_br | `lener_br` | — | train | ✓ | ✓ |
| macmorpho | `mac_morpho` | — | train | | ✓ |
| tweetsentbr | `eduagarcia/tweetsentbr_fewshot` | — | test | ✓ | ✓ |
| hatebr | `ruanchaves/hatebr` | — | train | ✓ | ✓ |
| pira | `paulopirozelli/pira` | — | train | ✓ | ✓ |

Cada tarefa é tolerante ao schema e **pulada com aviso** se o dataset não carregar
(um ID errado não derruba o build). IDs sobreponíveis com `--source nome=hf_id`.

## Reproduzir os dados | Reproduce the data

```bash
# ---------- v1 ----------
python sft/jaster/build_jaster.py --tasks default --out data/sft/jaster --shuffle
python sft/prepare_data.py --sources alpaca,oasst --out data/sft \
  --shuffle --dedup --extra data/sft/jaster/manaca_jaster.jsonl
# (ou simplesmente: make sft-data)

# ---------- v2 ----------
python sft/jaster/build_jaster.py --tasks all --out data/sft_v2/jaster \
  --shuffle --max_chars 5000 --max_per_task 8000
python sft/prepare_data.py --sources alpaca,aya,oasst,translation,summarization \
  --out data/sft_v2 --shuffle --dedup --max_chars 5000 \
  --extra data/sft_v2/jaster/manaca_jaster.jsonl
# (ou simplesmente: make sft-data-v2)
```

## Nota de licença | License note

O modelo **base** é CC BY 4.0. Já a mistura de **SFT** inclui fontes
não-comerciais (Alpaca-PT, linhagem CC BY-NC; XLSum, CC BY-NC-SA). Para uso
comercial do instruct, verifique a licença de cada fonte na sua ficha do Hugging
Face e, se preciso, refaça a mistura só com fontes permissivas (o pipeline aceita
`--sources` reduzido). As licenças acima são o melhor conhecimento no momento e
devem ser confirmadas na origem.
