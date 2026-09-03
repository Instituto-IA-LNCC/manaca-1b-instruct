---
language:
- pt
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: text-generation
base_model: menezesbruno/manaca-1b-base
tags:
- portuguese
- brazilian-portuguese
- instruction-tuned
- safety-alignment
- llama
- lncc
datasets:
- dominguesm/alpaca-data-pt-br
- CohereForAI/aya_dataset
- OpenAssistant/oasst1
- Helsinki-NLP/opus-100
- csebuetnlp/xlsum
---

# Manacá-1B-Instruct — instruct aberto e auditável para o Português do Brasil<br>An Open, Auditable Instruction-Tuned Brazilian-Portuguese Model

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Base model](https://img.shields.io/badge/%F0%9F%A4%97%20Base-menezesbruno%2Fmanaca--1b--base-yellow.svg)](https://huggingface.co/menezesbruno/manaca-1b-base)
[![Code & reproducibility](https://img.shields.io/badge/GitHub-manaca--1b--instruct-181717.svg)](https://github.com/brunoleomenezes/manaca-1b-instruct)
[![Language: PT-BR](https://img.shields.io/badge/Language-PT--BR-009c3b.svg)]()
[![Institution: LNCC](https://img.shields.io/badge/Institution-LNCC-002776.svg)](https://www.lncc.br)

<p align="center">
  <img src="assets/figures/manaca-identity.svg" width="520" alt="Manacá — Tibouchina mutabilis: os três estágios florais como metáfora do treinamento do LLM"/>
</p>

*Se o base é a flor branca (pré-treino), o instruct é o estágio púrpura: o alinhamento,*
*quando o Manacá aprende a ajudar com segurança, em Português do Brasil.*
<br>
*If the base is the white flower (pretraining), the instruct is the purple stage:*
*alignment, where Manacá learns to help safely, in Brazilian Portuguese.*

**[🇧🇷 Português](#-português) · [🇬🇧 English](#-english)**

---

## 🇧🇷 Português

**Manacá-1B-Instruct** é a versão **instruct, alinhada para segurança**, do
[Manacá-1B base](https://huggingface.co/menezesbruno/manaca-1b-base): um modelo
decoder-only de **~1,72 bilhão de parâmetros**, treinado do zero para o português do
Brasil, agora ajustado para **seguir instruções e recusar pedidos nocivos com
segurança**. Cooperação científica **LNCC × NII/LLM-jp**.

Todo o caminho é **reprodutível e auditável** (corpus, código, logs, avaliação — e
inclusive o que **não** funcionou): https://github.com/brunoleomenezes/manaca-1b-instruct

### Como usar

Template **Alpaca-PT** (o mesmo do treino — usar esse formato é importante):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

mid = "menezesbruno/manaca-1b-instruct"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="auto")

def responder(instrucao, entrada=""):
    preambulo = ("Abaixo está uma instrução que descreve uma tarefa. "
                 "Escreva uma resposta que atenda adequadamente ao pedido.")
    prompt = f"{preambulo}\n\n### Instrução:\n{instrucao}\n\n### Resposta:\n"
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=384, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print(responder("Explique o que é fotossíntese em uma frase."))
```

### O modelo

Herda a arquitetura do base (o pós-treino é um ajuste por cima desses pesos).

| Item | Valor |
|------|-------|
| Parâmetros | 1.722.951.680 (~1,72B) |
| Camadas / dim / FFN | 24 / 2048 / 8192 (SwiGLU) |
| Cabeças / grupos KV / head dim | 32 / 8 (GQA) / 64 |
| Posição / norma | RoPE (θ=500000) / RMSNorm |
| Contexto / precisão | 4096 / bfloat16 |
| Tokenizador | SentencePiece unigram, 64k, `nmt_nfkc_cf` |
| Template de conversa | Alpaca-PT (`### Instrução:` / `### Resposta:`) |
| Pós-treino | SFT (full FT) + safety-SFT (LoRA) |

### Como foi treinado

1. **SFT** (supervisionado), template Alpaca-PT, fiel ao `llm-jp-sft`. Duas versões
   de dados (v1 e v2, composições diferentes); a v2 é a base do modelo alinhado.
2. **Alinhamento de segurança (safety-SFT, estilo AnswerCarefully do LLM-jp):**
   recusa de pedidos nocivos como alvo de cross-entropy + exemplos de ajudar o
   benigno + dados gerais, para não virar recusador cego.

Resultado negativo documentado: o **DPO on-policy foi testado e não mudou a geração**
(nem nos próprios prompts de treino); a segurança veio do safety-SFT. Corpus (fontes,
splits, licenças, contagens), código e logs completos no GitHub.

### Resultados

Contra outros instructs PT-BR, mesmo harness (só instruct; o base não entra).

**MT-Bench-PT** (nota do juiz LLM, 1 a 10, por categoria):

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

**IFEval-PT** (seguir instrução verificável, acurácia %):

| Modelo | Prompt strict | Instr strict | Instr loose |
|---|:--:|:--:|:--:|
| manaca-instruct-v1 | 22.2 | 34.0 | 34.0 |
| manaca-instruct-v2 | 22.2 | 36.0 | 36.0 |
| **manaca-instruct-safe** | 25.0 | 36.0 | **36.0** |
| tucano-1b1-instruct | 11.1 | 22.0 | 26.0 |
| tucano-2b4-instruct | 13.9 | 26.0 | 26.0 |
| ttl-460m-chat | 8.3 | 18.0 | 18.0 |

**Bateria PT-BR — classificação (f1-macro %; acc entre parênteses):**

| Modelo | assin2_rte | faquad_nli | hatebr | hatespeech |
|---|:--:|:--:|:--:|:--:|
| manaca-instruct-v1 | 52.49 (53.6) | 43.92 (78.3) | 39.02 (50.0) | 41.74 (68.3) |
| manaca-instruct-v2 | 51.83 (51.8) | 44.93 (77.4) | 40.13 (48.7) | 42.29 (68.0) |
| **manaca-instruct-safe** | 51.54 (52.3) | 45.61 (77.5) | 40.59 (49.1) | 42.93 (67.9) |
| tucano-1b1-instruct | 35.69 (50.1) | 44.97 (71.2) | 30.64 (38.0) | 42.18 (66.2) |
| tucano-2b4-instruct | 43.75 (51.7) | 45.66 (74.9) | 37.19 (45.9) | 41.58 (68.2) |
| ttl-460m-chat | 53.61 (55.0) | 37.59 (38.2) | 40.67 (45.8) | 39.37 (39.9) |

**Bateria PT-BR — exames de múltipla escolha (acurácia %):**

| Modelo | enem | bluex | oab |
|---|:--:|:--:|:--:|
| manaca-instruct-v1 | 20.82 | 23.55 | 23.57 |
| manaca-instruct-v2 | 20.55 | 22.16 | 22.71 |
| **manaca-instruct-safe** | 19.50 | 21.19 | 22.90 |
| tucano-1b1-instruct | 20.06 | 24.65 | 26.65 |
| tucano-2b4-instruct | 20.82 | 20.78 | 23.98 |
| ttl-460m-chat | 19.36 | 23.96 | 24.62 |

**Segurança (sondas PT-BR, disjuntas do treino):**

| Métrica | instruct (SFT) | **instruct-safe** |
|---|:--:|:--:|
| Recusa de pedidos nocivos held-out (↑) | 0% | **75%** |
| MT-Bench-PT segurança (1-10) | 1.5 | **9.0** |
| Over-refusal em pedidos benignos (↓) | 2.9% | 8.8% |

Leitura: lideramos MT-Bench geral e IFEval-PT entre os instructs PT-BR; o `-safe`
domina segurança (recusa de nocivos 0→75%, nota 1,5→9,0) **sem** degradar IFEval nem
o leaderboard (assin2_rte f1 51,5 ≈ v2 51,8). Exames de múltipla escolha ficam no
acaso para todos os modelos dessa escala.

### Segurança e limitações

- **Recall de segurança ~75%**: pedidos de fraseado ambíguo podem passar.
- **Over-refusal**: uma fração pequena de pedidos benignos é recusada.
- **Caso pró-social**: pode recusar pedidos de *ajudar alguém em crise* (ex.: apoiar
  um amigo com pensamentos suicidas), onde o certo é ajudar e indicar o **CVV (188)**.
- Modelo de pesquisa (~1,72B): erra fatos e **não** substitui aconselhamento
  profissional, jurídico ou médico.

### Licença

Os pesos derivam do base (CC BY 4.0), mas a mistura de **SFT inclui fontes
não-comerciais** (Alpaca-PT, CC BY-NC; XLSum, CC BY-NC-SA). Por isso o card declara
**CC BY-NC 4.0**. Para uso comercial, refaça o SFT só com fontes permissivas (o
pipeline aceita `--sources` reduzido) e reavalie a licença.

### Citação

```
Menezes, B.L.S., Cardoso, C.L.S., & Porto, F.A.M. (2026). Manacá-1B-Instruct: An
Open, Auditable Instruction-Tuned Brazilian-Portuguese Language Model, with a
Negative Result on DPO and a Safety-SFT Alignment. Preprint. LNCC (Instituto de IA)
× NII/LLM-jp. https://github.com/brunoleomenezes/manaca-1b-instruct · CC BY-NC 4.0.
```

**Equipe:** Bruno Leonardo Santos Menezes · Carlos Leonardo Souza Cardoso ·
Prof. Fábio André Machado Porto — LNCC / Instituto de IA
(`brunolsm@lncc.br` · `cardoso@lncc.br` · `fporto@lncc.br`)

---

## 🇬🇧 English

**Manacá-1B-Instruct** is the **instruction-tuned, safety-aligned** version of
[Manacá-1B base](https://huggingface.co/menezesbruno/manaca-1b-base): a decoder-only
model of **~1.72B parameters**, trained from scratch for Brazilian Portuguese, now
tuned to **follow instructions and refuse harmful requests safely**. Scientific
cooperation **LNCC × NII/LLM-jp**. The whole path is **reproducible and auditable**
(corpus, code, logs, evaluation, and even what did **not** work):
https://github.com/brunoleomenezes/manaca-1b-instruct

### How to use

Alpaca-PT template (same as training) — see the Python snippet in the Portuguese
section above; just change the instruction text.

### Training

Post-training on Manacá-1B base: **(1) SFT** (Alpaca-PT, faithful to `llm-jp-sft`;
data v1/v2) then **(2) safety-SFT** (AnswerCarefully-style: refusal as a
cross-entropy target + help-the-benign + general data). Documented negative result:
on-policy **DPO did not change generation**; safety came from safety-SFT.

### Results

Same tables as the Portuguese section: **MT-Bench-PT** overall **2.90** (vs 2.50 best
PT-BR peer) and safety **9.0**; **IFEval-PT** instr-loose **36.0** (best among
PT-BR instructs); **PT-BR leaderboard** f1-macro/exams ≈ the v2 base (safety
alignment did not degrade measured capabilities). Held-out harmful-refusal rose
**0% → ~75%** with only ~9% over-refusal on benign prompts.

### Safety & limitations

Safety recall ~75% (ambiguous phrasings can pass); small over-refusal on benign
requests; may over-refuse *help-in-crisis* prompts (point users to a crisis line —
in Brazil, CVV 188). Research model (~1.72B): makes factual errors; **not** a
substitute for professional, legal, or medical advice.

### License & citation

CC BY-NC 4.0 (the SFT mix includes non-commercial sources: Alpaca-PT, XLSum). See the
citation block above.

*Projeto Manacá — LNCC (Instituto de IA) × NII/LLM-jp.*
