# Avaliação comparativa dos instructs Manacá vs. instructs PT-BR<br>Comparative evaluation of Manacá instructs vs. PT-BR instructs

Comparação dos **nossos instructs** (`manaca-instruct-v1`, `-v2` e o alinhado
`-safe`) contra outros modelos **instruct** de português, todos sob o mesmo
protocolo e harness. Aqui só entram modelos **instruct** (o base não entra nesta
comparação). Só resultados de corridas bem-sucedidas, com código e evidência
versionados para auditoria e reprodução.

Comparison of **our instructs** against other Portuguese **instruction-tuned**
models, same protocol and harness. Instruct models only (the base is out of this
comparison). Only successful runs, with code and evidence versioned.

Modelos de referência | Baselines: `tucano-1b1-instruct`, `tucano-2b4-instruct`
(TucanoBR), `ttl-460m-chat` (TeenyTinyLlama).

---

## 1. MT-Bench-PT (juiz LLM, 1–10)

Nota média por categoria (± erro padrão). Juiz: LLM-as-a-Judge (claude-opus-5),
avaliação single-answer, temperatura por categoria (0 objetivas / 0.7 criativas),
rubrica de segurança invertida (recusar = nota alta).

| Categoria | inst-v1 | inst-v2 | **inst-safe** | tucano-1b1-inst | tucano-2b4-inst | ttl-460m-chat |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| escrita | 2.83 | 2.50 | 2.67 | 2.50 | 3.17 | 1.17 |
| interpretação | 1.67 | 2.33 | 1.17 | 2.00 | 2.50 | 1.00 |
| raciocínio | 1.50 | 1.50 | 1.67 | 1.17 | 1.17 | 1.00 |
| matemática | 1.00 | 1.00 | 1.00 | 1.00 | 1.33 | 1.00 |
| código | 1.00 | 1.00 | 1.00 | 1.17 | 1.33 | 1.00 |
| extração | 3.33 | 4.33 | 4.33 | 3.33 | 3.33 | 3.33 |
| ciências | 3.50 | 3.83 | 2.50 | 1.83 | 3.33 | 2.00 |
| humanas | 4.00 | 4.67 | 4.83 | 4.50 | 4.83 | 1.17 |
| **segurança** | 1.50 | 1.50 | **9.00** | 1.50 | 1.00 | 5.00 |
| **GERAL** | 2.29 | 2.56 | **2.90** | 2.13 | 2.50 | 1.73 |

O `manaca-instruct-safe` lidera o **GERAL** (2.90) e domina **segurança** (9.0, o
único a recusar pedidos nocivos de forma consistente). Mesmo sem alinhamento, o
`manaca-instruct-v2` (2.56) supera o `tucano-2b4-instruct` (2.50), que é maior.

## 2. IFEval-PT (seguir instrução verificável)

Acurácia (%) ± erro padrão; instr-loose (nível de instrução, normalizado) é a
métrica principal.

| Modelo | Prompt strict | Instr strict | **Instr loose** |
|---|:--:|:--:|:--:|
| manaca-instruct-v1 | 22.2 | 34.0 | 34.0 |
| manaca-instruct-v2 | 22.2 | 36.0 | **36.0** |
| **manaca-instruct-safe** | 25.0 | 36.0 | **36.0** |
| tucano-1b1-instruct | 11.1 | 22.0 | 26.0 |
| tucano-2b4-instruct | 13.9 | 26.0 | 26.0 |
| ttl-460m-chat | 8.3 | 18.0 | 18.0 |

Os instructs Manacá lideram o IFEval-PT com folga (instr-loose 34–36 vs. 18–26 dos
demais), e o alinhamento de segurança **não** custou capacidade de seguir instrução
(safe = v2 = 36.0).

## 3. Bateria PT-BR (Open PT-LLM Leaderboard, adaptada)

### 3.1 Classificação — f1-macro (%)

O número rigoroso para tarefas desbalanceadas (a acurácia infla pela classe
majoritária; acc entre parênteses só para referência). Recalculado dos `samples`
do lm-eval.

| Modelo | assin2_rte | faquad_nli | hatebr | hatespeech |
|---|:--:|:--:|:--:|:--:|
| manaca-instruct-v1 | 52.49 (53.6) | 43.92 (78.3) | 39.02 (50.0) | 41.74 (68.3) |
| manaca-instruct-v2 | 51.83 (51.8) | 44.93 (77.4) | 40.13 (48.7) | **42.29** (68.0) |
| ttl-460m-chat | **53.61** (55.0) | 37.59 (38.2) | **40.67** (45.8) | 39.37 (39.9) |
| tucano-1b1-instruct | 35.69 (50.1) | 44.97 (71.2) | 30.64 (38.0) | 42.18 (66.2) |
| tucano-2b4-instruct | 43.75 (51.7) | **45.66** (74.9) | 37.19 (45.9) | 41.58 (68.2) |

Leitura honesta: em `assin2_rte` (a tarefa que mais discrimina) os instructs Manacá
(52.5 / 51.8) ficam à frente dos dois Tucano-instruct (35.7 / 43.8); nas demais o
grupo empata dentro do ruído. As altas acurácias de `faquad_nli`/`hatespeech`
escondem viés de classe majoritária — por isso o f1-macro é o número reportado.

### 3.2 Exames de múltipla escolha — acurácia (%)

| Modelo | enem | bluex | oab |
|---|:--:|:--:|:--:|
| manaca-instruct-v1 | 20.82 | 23.55 | 23.57 |
| manaca-instruct-v2 | 20.55 | 22.16 | 22.71 |
| tucano-1b1-instruct | 20.06 | 24.65 | 26.65 |
| tucano-2b4-instruct | 20.82 | 20.78 | 23.98 |
| ttl-460m-chat | 19.36 | 23.96 | 24.62 |

Todos ficam na **faixa do acaso** (ENEM ~20%, OAB ~25%) — esperado para instructs
de 0.4–2.4B; nenhum resolve exame de múltipla escolha nessa escala.

## 4. Síntese | Summary

- **Melhor instruct nosso**: `manaca-instruct-v2` em capacidade geral (MT-Bench
  GERAL, IFEval) e `manaca-instruct-safe` quando se soma segurança (MT-Bench GERAL
  2.90 e segurança 9.0), sem perder o seguir-instrução.
- **Frente aos pares PT-BR**: lideramos IFEval-PT e MT-Bench GERAL; competitivos ou
  à frente em `assin2_rte` (f1-macro); exames no acaso para todos.
- `manaca-instruct-safe` não foi rodado na bateria do leaderboard (foco do
  alinhamento foi segurança/utilidade; ver o registro de alinhamento).

## 5. Metodologia e reprodução

- **MT-Bench-PT**: `bench/mtbench_pt/` (gerar respostas → `judge.py` → `report.py`).
- **IFEval-PT**: `bench/ifeval_pt/` (gerar respostas → `score.py`).
- **Bateria PT-BR**: `scripts/eval/run_lm_eval_pt.sh` + `lm_eval_tasks/*.yaml`;
  `merge_ptbench.py` (tabela acc) e `f1_ptbench.py` (f1-macro dos samples).
- Evidência versionada: `bench/mtbench_pt/{answers,judged}/`,
  `bench/ifeval_pt/answers/` (as saídas cruas dos modelos e as notas do juiz).

Todos os modelos foram medidos no mesmo texto/harness; erros-padrão nas tabelas
originais (`bench/*/*.md`). Só corridas bem-sucedidas entram aqui.
