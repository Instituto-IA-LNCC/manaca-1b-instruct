# Servindo o Manacá-1B-Instruct em llama.cpp / llama-server

Guia de referência para servir o modelo em produção sobre `llama.cpp`
(por exemplo, no Carcará / Santos Dumont). Cobre os dois problemas conhecidos
da conversão para GGUF e os parâmetros de geração recomendados, todos
respaldados pelos dados do próprio repositório.

> Escopo: este documento trata **apenas** do serving em `llama.cpp`/GGUF. Os
> números do artigo e do leaderboard foram obtidos com o tokenizador HF
> corrigido via `transformers` (não pelo GGUF) e **não** são afetados pelos
> itens abaixo. Se produção servir um GGUF mal convertido, porém, a saída fica
> **pior** do que a reportada no artigo.

---

## 0. Resumo (dois problemas independentes)

1. **Tokenizador (crítico).** O tokenizador do Manacá é SentencePiece
   **Unigram** com normalização `nmt_nfkc_cf` (NFKC + minúsculas). O
   `convert_hf_to_gguf.py`, para arquitetura Llama, grava o GGUF como
   SPM/BPE clássico (`tokenizer.ggml.model = "llama"`, tipo 1) e **não**
   grava o `precompiled_charsmap`. Sem ele, o llama.cpp não normaliza: a
   segmentação diverge do treino e prompts com maiúscula degradam ou
   travam.
2. **`chat_template` ausente.** O `tokenizer_config.json` não traz
   `chat_template`, então o modo chat cai num fallback (ChatML) que não bate
   com o formato Alpaca-PT do treino. Sintoma: respostas vazias ou a
   pergunta ecoada, só em modo chat.

Ambos têm correção definitiva no lado do repositório/GGUF; enquanto isso, há
contornos no lado do servidor.

Créditos do diagnóstico: discussão da comunidade no Hugging Face
(`menezesbruno/manaca-1b-instruct`, discussão #2, por Henriik2) e a conversão
GGUF corrigida em `sulfierry/manaca-1b-instruct-GGUF`, que mapeou também um
segundo detalhe (tratamento de `\n` entre `tokenizer.json` e `tokenizer.model`).

---

## 1. Verificar o GGUF que está em produção

Rode contra o arquivo servido:

```bash
./llama-cli -m manaca-1b-instruct.f16.gguf --verbose-prompt -p "teste" 2>&1 \
  | grep -E "tokenizer.ggml.model|init_tokenizer|precompiled_charsmap"
```

| Situação | `tokenizer.ggml.model` | `init_tokenizer` | `precompiled_charsmap` |
| --- | --- | --- | --- |
| **Quebrado** | `llama` | `type 1` | ausente |
| **Correto** | `t5` | `type 4` | `arr[u8,244410]` |

Teste funcional rápido (o quebrado trava/degrada com maiúscula; o correto não):

```bash
./llama-cli -m <gguf> --temp 0 -p "Qual a capital do Brasil?"
```

---

## 2. Conversão correta (Unigram / UGM)

O caminho padrão do `convert_hf_to_gguf.py` para arquitetura Llama chama
`_set_vocab_sentencepiece()` incondicionalmente. Para tokenizadores Unigram é
preciso o caminho UGM (o mesmo que a classe T5 já usa), que grava
`tokenizer_model = "t5"` **e** o `precompiled_charsmap`.

O patch completo e a análise estão na discussão #2 do repositório no Hugging
Face (e no repositório de GGUF do sulfierry, que cobre também o caso do `\n`).
Depois de aplicar:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git apply manaca-ugm-tokenizer-fix.patch   # patch da discussão / do sulfierry
pip install -r requirements/requirements-convert_hf_to_gguf.txt
python3 convert_hf_to_gguf.py /caminho/para/manaca-1b-instruct \
    --outfile manaca-1b-instruct.f16.gguf --outtype f16
```

Reconverta **do zero** e valide com a seção 1 (`t5` / `type 4` / charsmap
presente). "instrução" deve virar um único token, idêntico ao `transformers`.

> Trata-se de um bug genérico do `llama.cpp` (qualquer modelo Llama-arch com
> tokenizador SentencePiece Unigram), não uma peculiaridade do Manacá. Um PR
> upstream corrigiria isso para toda a comunidade.

---

## 3. `chat_template` (formato Alpaca-PT)

O modelo foi treinado no template Alpaca-PT do SFT/DPO. Sem `chat_template`,
o servidor monta o prompt errado.

### 3a. Correção definitiva (recomendada): embutir no repositório HF

Adicionar o campo `chat_template` ao `tokenizer_config.json` no repositório do
modelo. Assim toda ferramenta downstream acerta sozinha, inclusive o
`convert_hf_to_gguf`, que copia o template para o GGUF. O valor exato (uma
linha, escapado para JSON) está em [`chat_template.snippet.json`](./chat_template.snippet.json).

Verificado: em single-turn, com `add_generation_prompt=True`, o template
reproduz **exatamente** o `montar_prompt()` de `bench/mtbench_pt/gen_answers.py`.

### 3b. Contorno no servidor: `--chat-template-file`

Enquanto o repositório HF não é atualizado, passe o template em Jinja
diretamente ao servidor. O arquivo está versionado aqui:
[`alpaca-pt.jinja`](./alpaca-pt.jinja).

```bash
./llama-server -m manaca-1b-instruct.f16.gguf --chat-template-file docs/serving/alpaca-pt.jinja
```

> Ressalva: o Alpaca-PT foi treinado **single-turn** (uma instrução, uma
> resposta). O template estende para múltiplos turnos por conveniência, mas o
> modelo não viu esse padrão no treino; a qualidade tende a cair conforme a
> conversa cresce. Para máxima fidelidade, envie apenas o último turno do
> usuário.

---

## 4. Parâmetros de geração (respaldados por varredura no IFEval-PT)

Medição automática e objetiva com `scripts/eval/sweep_decoding.py` (pontua as
saídas com os próprios checadores do repositório, sem juiz; n = 50 instruções):

| Config | instr-loose |
| --- | ---: |
| greedy (temp 0) | 36.0% |
| temp 0.3 / top_p 0.9 | 36.0% |
| greedy + repeat_penalty 1.15 | 30.0% |
| temp 0.7 / top_p 0.9 | 30.0% |
| greedy + repeat_penalty 1.30 | 26.0% |

Leituras: greedy é o melhor e é reprodutível; `temp 0.3` empata sem custo;
`repeat_penalty` **piora** de forma monotônica; temperatura mais alta piora.
Com n = 50, diferenças de 1-2 pontos são ruído, mas o padrão (penalidade e
temperatura alta sempre iguais ou piores) é consistente.

### Preset `manaca-eval` (reprodutível: leaderboard, tarefas objetivas)

```json
{ "temperature": 0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
  "repeat_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0, "seed": 0 }
```

### Preset `manaca-chat` (uso interativo: menos repetição, sem perder aderência)

```json
{ "temperature": 0.3, "top_p": 0.9, "min_p": 0.0,
  "repeat_penalty": 1.0,
  "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2, "dry_penalty_last_n": -1 }
```

- Prefira o sampler **DRY** (penaliza sequências repetidas) a `repeat_penalty`
  (penaliza tokens individuais e derruba o seguimento de instrução).
- **Sete os parâmetros explicitamente**; o default do `repeat_penalty` variou
  entre versões do `llama.cpp` — não confie nele.

### Parada e comprimento (ambos os presets)

```json
"stop": ["\n### Instrução:", "\n### Resposta:", "</s>"],
"max_tokens": 768
```

`max_tokens`/`n_predict` é o teto de tokens da resposta. 768 = igual ao bench;
512 é seguro e mais econômico (as respostas terminam no `</s>` bem antes).
As stop-strings evitam o modelo "recomeçar" um turno Alpaca (sintoma que
parece repetição, mas é falta de parada).

---

## 5. Precisão numérica

A varredura acima foi em bf16 (`transformers`). Para a saída bater com o
artigo/HF, sirva **F16**. Quantização (Q4/Q5/Q8) economiza VRAM mas altera o
argmax e pode piorar respostas curtas/objetivas. Ordem por qualidade:
`F16 > Q8_0 > Q5_K_M > Q4`. Em uma H100, F16 de um modelo ~1B cabe folgado.

---

## 6. Comportamentos esperados (não são bugs)

- **Saída em minúsculas.** O treino usa case folding (`nmt_nfkc_cf`); o modelo
  gera minúsculo por construção. Não "corrigir" com pós-processamento que
  altere conteúdo.
- **Modelo single-turn.** Conversas longas degradam; é intrínseco a um modelo
  ~1B treinado single-turn, não um defeito de configuração.
- **Alguma repetição em texto longo aberto** é intrínseca a modelos pequenos;
  `manaca-chat` (temp 0.3 + DRY + stop) a deixa residual sem machucar a
  coerência. Perseguir "zero repetição" com penalidade alta piora mais do que
  ajuda.

---

## Ordem prática de checagem para o dev

1. Rodar a seção 1 no GGUF de produção.
2. Se `llama`/`type 1`/sem charsmap: reconverter com o patch UGM (seção 2).
3. Garantir o `chat_template` (seção 3a no repo, ou 3b como contorno).
4. Aplicar o preset de sampling adequado ao endpoint (seção 4).
5. Preferir F16 (seção 5).
