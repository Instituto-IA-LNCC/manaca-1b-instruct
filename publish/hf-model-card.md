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

# Manacá-1B-Instruct

Versão **instruct (alinhada para segurança)** do [Manacá-1B base](https://huggingface.co/menezesbruno/manaca-1b-base):
um modelo decoder-only de **~1,72 bilhão de parâmetros**, treinado do zero para o
**português do Brasil**, agora ajustado para **seguir instruções e recusar pedidos
nocivos com segurança**. Cooperação científica **LNCC × NII/LLM-jp**.

Instruction-tuned, safety-aligned version of Manacá-1B (base). ~1.72B params,
Brazilian Portuguese, LNCC × NII/LLM-jp.

**Pipeline reprodutível e auditável (código, dados, logs, avaliação):**
https://github.com/brunoleomenezes/manaca-1b-instruct

## Como usar | How to use

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

## Treino | Training

Pós-treino sobre o Manacá-1B base:

1. **SFT** (fine-tuning supervisionado), template Alpaca-PT, fiel ao `llm-jp-sft`.
   Duas versões de dados (v1 e v2); a v2 é a base do modelo alinhado.
2. **Alinhamento de segurança (safety-SFT, estilo AnswerCarefully do LLM-jp):**
   recusa de pedidos nocivos como alvo de cross-entropy + exemplos de ajudar o
   benigno + dados gerais, para não virar recusador cego.

Nota metodológica: o **DPO on-policy foi testado e não mudou a geração** (resultado
negativo documentado no repositório); a segurança veio do safety-SFT.

Corpus, código e logs completos no repositório do GitHub.

## Avaliação | Evaluation

Contra outros instructs PT-BR, mesmo harness (só instruct). Detalhes e erros-padrão
no repositório.

| Métrica | Manacá-instruct-safe | melhor par PT-BR |
|---|:--:|:--:|
| MT-Bench-PT (GERAL, 1-10, juiz LLM) | **2.90** | 2.50 (Tucano-2b4-inst) |
| MT-Bench-PT segurança (1-10) | **9.0** | 5.0 (TTL-460m-chat) |
| IFEval-PT (instr-loose, %) | **36.0** | 26.0 (Tucano-inst) |
| assin2-rte (f1-macro, %) | 51.5 | 53.6 (TTL-460m-chat) |

Segurança (sondas PT-BR, disjuntas do treino): recusa de pedidos nocivos inéditos
subiu de **0% → ~75%**; over-refusal em pedidos benignos ~9% (o alinhamento não
regrediu IFEval nem o leaderboard PT-BR).

## Segurança e limitações | Safety & limitations

- **Recall de segurança ~75%**: pedidos de fraseado ambíguo podem passar.
- **Over-refusal**: uma fração pequena de pedidos benignos é recusada.
- **Caso pró-social**: pode recusar pedidos de *ajudar alguém em crise* (ex.: apoiar
  um amigo com pensamentos suicidas), onde o certo é ajudar e indicar o **CVV (188)**.
- Modelo de pesquisa (~1,72B): erra fatos, não resolve exames de múltipla escolha
  (fica no acaso) e **não** substitui aconselhamento profissional, jurídico ou médico.

## Licença | License

Os pesos derivam do Manacá-1B base (CC BY 4.0), mas a mistura de **SFT inclui fontes
não-comerciais** (Alpaca-PT, linhagem CC BY-NC; XLSum, CC BY-NC-SA). Por isso o card
declara **CC BY-NC 4.0** por precaução. Para uso comercial, refaça o SFT só com
fontes permissivas (o pipeline aceita `--sources` reduzido) e reavalie a licença.

## Citação | Citation

```
Menezes, B.L.S., Cardoso, C.L.S., & Porto, F.A.M. (2026). Manacá-1B-Instruct: An
Open, Auditable Instruction-Tuned Brazilian-Portuguese Language Model, with a
Negative Result on DPO and a Safety-SFT Alignment. Preprint. LNCC (Instituto de IA)
× NII/LLM-jp. https://github.com/brunoleomenezes/manaca-1b-instruct · CC BY-NC 4.0.
```

*Projeto Manacá — LNCC (Instituto de IA) × NII/LLM-jp.*
