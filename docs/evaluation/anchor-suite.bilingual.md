# Suíte de Avaliação Âncora — Manacá / Anchor Evaluation Suite

**Versão / Version:** v1 — 2026-09

---

# 🇧🇷 Português

## Propósito

Esta é a suíte de avaliação **congelada** da família Manacá: o conjunto exato de testes
(prompts, tamanhos e código de pontuação) carregado **sem alteração** entre escalas de
modelo, do Manacá-1B (1,72B) ao ~10B em desenvolvimento, para que a fronteira
recusa×utilidade e os números de capacidade sejam **comparáveis ponto a ponto**.
Amplitude de cenários é bem-vinda, mas entra como **camada aditiva** reportada à parte,
nunca editando a âncora. O princípio: *amplitude sem uma suíte fixa só redesenha o
painel*, não mede uma mudança real entre escalas.

## Motivação

O resultado negativo do DPO (Seção 8 do artigo) mostrou que uma **métrica escalar de
treino não certifica comportamento**: a margem de recompensa saturou em ~1,0 e a acurácia
de ranking em ~1,0 enquanto a taxa de recusa em geração livre (greedy) ficou em 0/16, até
nos próprios prompts de treino. A âncora existe para que o comportamento seja medido em
**gerações held-out**, de forma reprodutível e comparável entre escalas, e não por um
número de treino.

## Componentes congelados

Caminhos relativos à raiz do repositório. Pontuação pelos scripts do próprio repositório.

### A. Fronteira de alinhamento / segurança

| Componente | N (congelado) | Arquivo | Mede |
|---|---:|---|---|
| Probes nocivos held-out | 16 | `bench/safety_pt/heldout.jsonl` | Taxa de **recusa** (deve recusar) |
| Probes de over-refusal | 34 | `bench/overrefusal_pt/prompts.jsonl` | Taxa de **cumprimento** (não deve recusar), lado da utilidade |
| MT-Bench-PT | 54 perguntas · 9 categorias (segurança separável) | `bench/mtbench_pt/mtbench-pt.json` + `judge.py` | Qualidade por categoria (média + SE) |
| IFEval-PT | 36 prompts · 50 instruções verificáveis | `bench/ifeval_pt/prompts.jsonl` + `checkers.py` / `score.py` | Seguimento de instrução |

A **matriz recusa/utilidade** é a leitura conjunta dos dois primeiros: recusa em 16 probes
nocivos (subir é bom) vs cumprimento em 34 probes benignos que parecem nocivos (recusar é
ruim). Reporta-se a **fronteira** (as duas taxas), nunca um escalar único; a categoria de
segurança do MT-Bench é sempre mantida **separável** do agregado de capacidade.

### B. Capacidade (lm-eval PT, log-verossimilhança, invariante ao tokenizador)

| Componente | Registro congelado |
|---|---|
| CALAME-PT, LAMBADA-PT, ARC-PT, HellaSwag-PT | `docs/evaluation/benchmarks-pt.json` |
| Exames: ENEM, BLUEX, OAB | `docs/evaluation/lmeval_ptbench/` |

Os tamanhos de cada split são os padrões do harness e ficam fixados no registro acima.
Como são métricas de log-verossimilhança, são **invariantes ao tokenizador**.

## Manifesto e traces

Para que cada ganho tenha **causa auditável** (separando drift de modelo de drift de
harness), publicamos, ao lado dos agregados:

- **Manifestos (versionados):** os conjuntos de prompts de cada componente
  (`bench/safety_pt/heldout.jsonl`, `bench/overrefusal_pt/prompts.jsonl`,
  `bench/ifeval_pt/prompts.jsonl`, `bench/mtbench_pt/mtbench-pt.json`) e os splits de
  capacidade em `docs/evaluation/benchmarks-pt.json`.
- **Traces crus por exemplo:** IFEval (`bench/ifeval_pt/answers/`), MT-Bench
  (`bench/mtbench_pt/answers/` + `judged/`), capacidade (`docs/evaluation/vectors-pt.json`,
  `docs/evaluation/logs/`, `docs/evaluation/lmeval_ptbench/**/samples_*.jsonl`).
- **Sondas de segurança (exceção deliberada):** manifestos, detector de recusa
  (`eh_recusa`), agregados e a fronteira ficam públicos em
  `docs/evaluation/safety-alignment-pt.md`; as **gerações cruas** das sondas nocivas
  **não** são commitadas por conterem conteúdo potencialmente danoso. Divulgação
  responsável, não opacidade.

## Regras da âncora (o contrato)

1. **Congelado.** Prompts, N e código de pontuação são fixados e carregados sem alteração
   de 1,72B → ~10B. Nada de reamostrar, retraduzir ou re-splittar em silêncio.
2. **Amplitude é aditiva.** Novos cenários entram como camada separada, reportada à parte;
   nunca editando ou removendo itens da âncora.
3. **Fronteira, não escalar.** Recusa×over-refusal é reportada como par (matriz/curva). A
   categoria de segurança do MT-Bench permanece separável.
4. **Decodificação.** Tarefas objetivas e pontuadas da âncora usam **greedy** com semente
   fixa (reprodutível); qualquer desvio de amostragem é documentado
   (ver `docs/serving/llama-server.md`, preset `manaca-eval`).
5. **Controle de mudança.** Qualquer alteração em um item da âncora **cria uma nova
   versão** da suíte; a versão anterior é preservada, jamais editada silenciosamente. Cada
   número reportado cita a versão da âncora usada.

## Histórico de versões

| Versão | Data | Mudanças |
|---|---|---|
| v1 | 2026-09 | Âncora inicial, congelada a partir da avaliação do Manacá-1B/1B-Instruct (probes 16+34, MT-Bench-PT 54, IFEval-PT 36/50, capacidade lm-eval PT + exames). |

## Nota

O princípio de **congelar o N de probes entre escalas** (para a fronteira permanecer
comparável, em vez de só redesenhar o painel) foi reforçado em discussão pública sobre o
lançamento, e consolida a metodologia de avaliação honesta e reprodutível do artigo.

---

# 🇬🇧 English

## Purpose

This is the Manacá family's **frozen** evaluation suite: the exact set of tests (prompts,
sizes, and scoring code) carried **unchanged** across model scales, from Manacá-1B (1.72B)
to the ~10B in development, so the refusal-vs-usefulness frontier and the capability
numbers stay **comparable point-for-point**. Breadth is welcome, but it enters as an
**additive layer** reported separately, never by editing the anchor. The principle:
*breadth without a fixed suite just redraws the dashboard*, it does not measure a real
shift across scales.

## Rationale

The DPO negative result (Section 8 of the paper) showed that a **scalar training metric
does not certify behavior**: the reward margin saturated at ~1.0 and ranking accuracy at
~1.0 while the free-generation (greedy) refusal rate stayed at 0/16, even on its own
training prompts. The anchor exists so that behavior is measured on **held-out
generations**, reproducibly and comparably across scales, not by a training number.

## Frozen components

Paths are relative to the repository root. Scoring uses the repository's own scripts.

### A. Alignment / safety frontier

| Component | N (frozen) | File | Measures |
|---|---:|---|---|
| Held-out harmful probes | 16 | `bench/safety_pt/heldout.jsonl` | **Refusal** rate (must refuse) |
| Over-refusal probes | 34 | `bench/overrefusal_pt/prompts.jsonl` | **Compliance** rate (must not refuse), usefulness side |
| MT-Bench-PT | 54 questions · 9 categories (safety separable) | `bench/mtbench_pt/mtbench-pt.json` + `judge.py` | Per-category quality (mean + SE) |
| IFEval-PT | 36 prompts · 50 verifiable instructions | `bench/ifeval_pt/prompts.jsonl` + `checkers.py` / `score.py` | Instruction following |

The **refusal/usefulness matrix** is the joint reading of the first two: refusal on 16
harmful probes (higher is better) vs compliance on 34 benign-but-unsafe-looking probes
(refusing is bad). Report the **frontier** (both rates), never a single scalar; the
MT-Bench safety category is always kept **separable** from the capability aggregate.

### B. Capability (lm-eval PT, log-likelihood, tokenizer-invariant)

| Component | Frozen record |
|---|---|
| CALAME-PT, LAMBADA-PT, ARC-PT, HellaSwag-PT | `docs/evaluation/benchmarks-pt.json` |
| Exams: ENEM, BLUEX, OAB | `docs/evaluation/lmeval_ptbench/` |

Each split's size follows the harness defaults and is pinned in the record above. Being
log-likelihood metrics, they are **tokenizer-invariant**.

## Manifests and raw traces

So that every later gain has an **auditable cause** (separating model drift from harness
drift), we publish, beside the aggregates:

- **Manifests (versioned):** the prompt sets for each component
  (`bench/safety_pt/heldout.jsonl`, `bench/overrefusal_pt/prompts.jsonl`,
  `bench/ifeval_pt/prompts.jsonl`, `bench/mtbench_pt/mtbench-pt.json`) and the capability
  splits in `docs/evaluation/benchmarks-pt.json`.
- **Raw per-example traces:** IFEval (`bench/ifeval_pt/answers/`), MT-Bench
  (`bench/mtbench_pt/answers/` + `judged/`), capability (`docs/evaluation/vectors-pt.json`,
  `docs/evaluation/logs/`, `docs/evaluation/lmeval_ptbench/**/samples_*.jsonl`).
- **Safety probes (deliberate exception):** manifests, refusal detector (`eh_recusa`),
  aggregates and the frontier are public in `docs/evaluation/safety-alignment-pt.md`; the
  **raw generations** of the harmful probes are **not** committed because they can contain
  harmful content. Responsible disclosure, not opacity.

## Anchor rules (the contract)

1. **Frozen.** Prompts, N, and scoring code are pinned and carried unchanged from 1.72B →
   ~10B. No silent resampling, retranslation, or re-splitting.
2. **Breadth is additive.** New scenarios enter as a separate layer, reported separately;
   never by editing or removing anchor items.
3. **Frontier, not a scalar.** Refusal vs over-refusal is reported as a pair
   (matrix/curve). The MT-Bench safety category stays separable.
4. **Decoding.** Objective, scored anchor tasks use **greedy** with a fixed seed
   (reproducible); any sampling deviation is documented
   (see `docs/serving/llama-server.md`, `manaca-eval` preset).
5. **Change control.** Any change to an anchor item **creates a new suite version**; the
   previous version is retained, never silently edited. Every reported number cites the
   anchor version used.

## Version history

| Version | Date | Changes |
|---|---|---|
| v1 | 2026-09 | Initial anchor, frozen from the Manacá-1B / 1B-Instruct evaluation (16+34 probes, MT-Bench-PT 54, IFEval-PT 36/50, lm-eval PT capability + exams). |

## Acknowledgment

The principle of **freezing the probe N across scales** (so the frontier stays comparable
rather than merely redrawing the dashboard) was reinforced in public discussion of the
release, and consolidates the honest, reproducible evaluation methodology of the paper.
