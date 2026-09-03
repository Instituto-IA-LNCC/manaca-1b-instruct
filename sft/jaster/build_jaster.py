#!/usr/bin/env python3
"""
Manacá-1B — Fase 3a · manacá-jaster | Jaster PT-BR (conversão de tarefas de NLP)
================================================================================
PT
--
O **manacá-jaster** é o análogo em português do **Jaster** do llm-jp-sft: a
"metade tarefas de NLP" do SFT, obtida convertendo AUTOMATICAMENTE datasets de
NLP em PT-BR para o formato de instrução. Assim como o Jaster ensina o modelo a
seguir instruções curtas ("Qual a similaridade semântica?" -> "4.0") a partir de
datasets japoneses (JNLI, JSICK, JSQuAD...), o manacá-jaster faz o mesmo com
ASSIN2, SICK-BR, FaQuAD, ENEM, HAREM, LeNER-Br, Mac-Morpho, TweetSentBR, HateBR
e Pirá (análise: references/phase3-sft/llm-jp-sft-pipeline-analise.md §§57.2).

Saída: `{instruction, input, output}` (esquema Alpaca) — o MESMO consumido por
`sft/prepare_data.py` e `sft/train.py`. O preâmbulo e os marcadores
`### Instrução:` / `### Entrada:` / `### Resposta:` são aplicados pelo
`train.py`; aqui produzimos só os campos crus, para o mix passar pelo mesmo
caminho de formatação/mascaramento (loss só na resposta).

Cada tarefa é tolerante ao schema (tenta múltiplos nomes de campo) e é PULADA com
aviso se o dataset não carregar — um ID/config errado não derruba o build todo.
IDs padrão são sobreponíveis (`--source nome=hf_id[:config]`).

EN
--
`manacá-jaster` is the PT-BR analogue of llm-jp-sft's **Jaster**: the "NLP-tasks"
half of the SFT mix, built by automatically converting Portuguese NLP datasets
into instruction format. Emits Alpaca `{instruction, input, output}` (same schema
as prepare_data.py / train.py); the preamble and markers are applied by train.py.
Each task tolerates schema variation and is skipped (with a warning) if its
dataset fails to load.

Uso | Usage:
    python sft/jaster/build_jaster.py --tasks all --out data/sft/jaster --shuffle
    # subconjunto + override de ID/config:
    python sft/jaster/build_jaster.py --tasks assin2_sts,faquad,enem \\
        --source harem=arubenruben/HAREM-Default --out data/sft/jaster

Depois, junte ao mix no treino:
    python sft/prepare_data.py --sources bode,aya,oasst --out data/sft \\
        --extra data/sft/jaster/manaca_jaster.jsonl --shuffle --dedup

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

# ── Utilitários de schema tolerante ──────────────────────────────────────────


def first(ex, *keys, default=""):
    """Primeiro campo não-vazio dentre `keys` (schema tolerante)."""
    for k in keys:
        v = ex.get(k)
        if v not in (None, "", []):
            return v
    return default


def s(x):
    return ("" if x is None else str(x)).strip()


def feature_names(ds, column):
    """Nomes de classes (id2label) de uma coluna ClassLabel/Sequence(ClassLabel)."""
    try:
        feat = ds.features[column]
        inner = getattr(feat, "feature", feat)     # Sequence -> feature interna
        names = getattr(inner, "names", None)
        if names:
            return list(names)
    except Exception:
        pass
    return None


# ── NER / POS (tokens + tags -> texto de instrução) ──────────────────────────


def join_tokens(tokens):
    """Reconstrói o texto a partir de tokens (aproximação; cola pontuação)."""
    out = ""
    for t in tokens:
        if t in {",", ".", ";", ":", "!", "?", ")", "]", "}", "%"} or (out and out[-1] in "([{"):
            out += t
        elif not out:
            out = t
        else:
            out += " " + t
    return out


def bio_entities(tokens, tag_ids, names):
    """Extrai entidades de rótulos BIO (B-/I-/O). Retorna [(tipo, superfície)]."""
    ents, cur_type, cur = [], None, []

    def flush():
        nonlocal cur_type, cur
        if cur_type and cur:
            ents.append((cur_type, join_tokens(cur)))
        cur_type, cur = None, []

    for tok, tid in zip(tokens, tag_ids):
        tag = names[tid] if names and 0 <= int(tid) < len(names) else str(tid)
        if tag in ("O", "0") or tag.upper() == "O":
            flush(); continue
        prefix, _, etype = tag.partition("-")
        etype = etype or prefix
        if prefix == "B" or not cur or (prefix == "I" and etype != cur_type):
            flush(); cur_type, cur = etype, [tok]
        else:
            cur.append(tok)
    flush()
    return ents


def fmt_entities(ents):
    if not ents:
        return "Nenhuma entidade nomeada encontrada."
    by_type = {}
    for etype, surf in ents:
        by_type.setdefault(etype, []).append(surf)
    parts = [f"{etype}: {', '.join(dict.fromkeys(v))}" for etype, v in by_type.items()]
    return "; ".join(parts)


# ── Conversores por tarefa (geradores -> {instruction, input, output}) ────────
# Preâmbulo/marcadores ficam a cargo do train.py; aqui só os campos crus.


def conv_assin2_sts(ds):
    inst = "Avalie a similaridade semântica entre as duas frases numa escala de 1 a 5."
    for ex in ds:
        a, b = s(first(ex, "premise", "sentence_A", "sentenca1")), s(first(ex, "hypothesis", "sentence_B", "sentenca2"))
        score = first(ex, "relatedness_score", "similarity", default=None)
        if not a or not b or score in (None, ""):
            continue
        try:
            out = f"{float(score):.1f}"
        except (TypeError, ValueError):
            out = s(score)
        yield {"instruction": inst, "input": f"Frase 1: {a}\nFrase 2: {b}", "output": out}


def _nli_label_pt(raw, names):
    """Normaliza rótulo de NLI para PT (Implicação / Neutro / Contradição / Nenhuma)."""
    if names is not None:
        try:
            raw = names[int(raw)]
        except (TypeError, ValueError, IndexError):
            pass
    key = s(raw).upper()
    return {
        "ENTAILMENT": "Implicação", "1": "Implicação", "IMPLICAÇÃO": "Implicação",
        "NEUTRAL": "Neutro", "NEUTRO": "Neutro",
        "CONTRADICTION": "Contradição", "CONTRADIÇÃO": "Contradição",
        "NONE": "Nenhuma", "0": "Nenhuma", "NENHUMA": "Nenhuma",
    }.get(key, s(raw))


def conv_assin2_nli(ds):
    names = feature_names(ds, "entailment_judgment")
    inst = ("Determine a relação lógica entre a premissa e a hipótese. "
            "Responda com uma palavra: Implicação ou Nenhuma.")
    for ex in ds:
        a, b = s(first(ex, "premise")), s(first(ex, "hypothesis"))
        lab = first(ex, "entailment_judgment", "entailment", "label", default=None)
        if not a or not b or lab in (None, ""):
            continue
        yield {"instruction": inst, "input": f"Premissa: {a}\nHipótese: {b}",
               "output": _nli_label_pt(lab, names)}


def conv_sick_br(ds):
    names = feature_names(ds, "entailment_label") or feature_names(ds, "label")
    inst = ("Determine a relação lógica entre a premissa e a hipótese. "
            "Responda com uma palavra: Implicação, Neutro ou Contradição.")
    for ex in ds:
        a = s(first(ex, "sentence_A", "premise", "sentence_a"))
        b = s(first(ex, "sentence_B", "hypothesis", "sentence_b"))
        lab = first(ex, "entailment_label", "label", "entailment_AB", default=None)
        if not a or not b or lab in (None, ""):
            continue
        yield {"instruction": inst, "input": f"Premissa: {a}\nHipótese: {b}",
               "output": _nli_label_pt(lab, names)}


def conv_sick_br_sts(ds):
    inst = "Avalie a similaridade semântica entre as duas frases numa escala de 1 a 5."
    for ex in ds:
        a = s(first(ex, "sentence_A", "premise", "sentence_a"))
        b = s(first(ex, "sentence_B", "hypothesis", "sentence_b"))
        score = first(ex, "relatedness_score", "similarity", default=None)
        if not a or not b or score in (None, ""):
            continue
        try:
            out = f"{float(score):.1f}"
        except (TypeError, ValueError):
            continue
        yield {"instruction": inst, "input": f"Frase 1: {a}\nFrase 2: {b}", "output": out}


def conv_faquad(ds):
    inst = ("Leia o contexto e responda à pergunta de forma extrativa, "
            "usando apenas informação presente no texto.")
    for ex in ds:
        ctx, q = s(first(ex, "context", "contexto")), s(first(ex, "question", "pergunta"))
        ans = first(ex, "answers", "answer", "resposta", default=None)
        if isinstance(ans, dict):
            txt = ans.get("text") or ans.get("answer_text") or []
            ans = txt[0] if isinstance(txt, list) and txt else (txt if isinstance(txt, str) else "")
        elif isinstance(ans, list):
            ans = ans[0] if ans else ""
        ans = s(ans)
        if not ctx or not q or not ans:
            continue
        yield {"instruction": inst, "input": f"Contexto: {ctx}\n\nPergunta: {q}", "output": ans}


def conv_enem(ds):
    inst = ("Responda à questão de múltipla escolha a seguir indicando "
            "a letra da alternativa correta (A, B, C, D ou E).")
    letters = ["A", "B", "C", "D", "E"]
    for ex in ds:
        header = s(first(ex, "header", "context", default=""))
        stmt = s(first(ex, "question", "statement", "enunciado", default=""))
        q = (header + "\n" + stmt).strip() if header else stmt
        alts = first(ex, "alternatives", "choices", "options", default=None)
        # alternatives pode ser lista de strings, ou dict {text:[...], label:[...]}
        alt_texts = None
        if isinstance(alts, dict):
            alt_texts = alts.get("text") or alts.get("choices")
        elif isinstance(alts, list):
            alt_texts = alts
        if not q or not alt_texts:
            continue
        lines = []
        for i, a in enumerate(alt_texts):
            a = s(a)
            # remove prefixo "A)"/"A." já presente para re-rotular de forma uniforme
            for pre in (f"{letters[i]})", f"{letters[i]}.", f"{letters[i]} -", f"{letters[i]}-"):
                if a.upper().startswith(pre):
                    a = a[len(pre):].strip(); break
            lines.append(f"{letters[i]}) {a}")
        key = s(first(ex, "label", "answerKey", "answer", "gabarito", default=""))
        if key.isdigit():
            idx = int(key)
            key = letters[idx] if 0 <= idx < len(letters) else key
        key = key.upper()[:1]
        if key not in letters:
            continue
        yield {"instruction": inst, "input": f"{q}\n\n" + "\n".join(lines),
               "output": f"Alternativa correta: {key}"}


def _conv_ner(column, task_desc):
    def conv(ds):
        names = feature_names(ds, column)
        inst = task_desc
        for ex in ds:
            toks = first(ex, "tokens", "words", default=None)
            tags = ex.get(column) if column in ex else first(ex, "ner_tags", "tags", "labels", default=None)
            if not toks or tags is None:
                continue
            text = join_tokens([s(t) for t in toks])
            ents = bio_entities([s(t) for t in toks], tags, names)
            yield {"instruction": inst, "input": text, "output": fmt_entities(ents)}
    return conv


def conv_macmorpho(ds):
    names = feature_names(ds, "pos_tags") or feature_names(ds, "tags")
    inst = ("Faça a etiquetagem morfossintática (POS) da frase, "
            "no formato palavra/ETIQUETA para cada palavra.")
    for ex in ds:
        toks = first(ex, "tokens", "words", default=None)
        tags = first(ex, "pos_tags", "tags", "labels", default=None)
        if not toks or tags is None:
            continue
        pairs = []
        for t, tid in zip(toks, tags):
            tag = names[int(tid)] if names and str(tid).lstrip("-").isdigit() and 0 <= int(tid) < len(names) else s(tid)
            pairs.append(f"{s(t)}/{tag}")
        yield {"instruction": inst, "input": join_tokens([s(t) for t in toks]),
               "output": " ".join(pairs)}


def conv_tweetsentbr(ds):
    names = feature_names(ds, "label") or feature_names(ds, "sentiment")
    inst = ("Classifique o sentimento da mensagem. "
            "Responda com uma palavra: Positivo, Neutro ou Negativo.")
    m = {"0": "Negativo", "1": "Neutro", "2": "Positivo",
         "NEGATIVE": "Negativo", "NEUTRAL": "Neutro", "POSITIVE": "Positivo",
         "NEGATIVO": "Negativo", "NEUTRO": "Neutro", "POSITIVO": "Positivo"}
    for ex in ds:
        text = s(first(ex, "sentence", "text", "tweet", "content"))
        lab = first(ex, "label", "sentiment", "polarity", default=None)
        if not text or lab in (None, ""):
            continue
        if names is not None:
            try:
                lab = names[int(lab)]
            except (TypeError, ValueError, IndexError):
                pass
        out = m.get(s(lab).upper(), s(lab))
        yield {"instruction": inst, "input": text, "output": out}


def conv_hatebr(ds):
    inst = ("A mensagem a seguir contém discurso de ódio ou linguagem ofensiva? "
            "Responda apenas com Sim ou Não.")
    for ex in ds:
        text = s(first(ex, "instagram_comments", "text", "comment", "sentence"))
        lab = first(ex, "offensive_language", "offensive", "hate_speech", "label", default=None)
        if not text or lab in (None, ""):
            continue
        val = s(lab).upper()
        pos = val in ("1", "TRUE", "SIM", "YES", "OFFENSIVE")
        yield {"instruction": inst, "input": text, "output": "Sim" if pos else "Não"}


def conv_pira(ds):
    inst = "Leia o contexto e responda à pergunta em português."
    for ex in ds:
        ctx = s(first(ex, "context_pt", "context_portuguese", "abstract", "context", default=""))
        q = s(first(ex, "question_pt_origin", "question_pt", "question_portuguese", "question_pt_paraphrase", "question"))
        ans = s(first(ex, "answer_pt_origin", "answer_pt", "answer_portuguese", "answer_pt_paraphrase", "answer"))
        if not q or not ans:
            continue
        inp = f"Contexto: {ctx}\n\nPergunta: {q}" if ctx else f"Pergunta: {q}"
        yield {"instruction": inst, "input": inp, "output": ans}


# ── Registro de tarefas (id HF padrão + config + split + conversor) ───────────
NER_HAREM = "Extraia as entidades nomeadas do texto, agrupadas por tipo."
NER_LENER = "Extraia as entidades nomeadas (jurídicas) do texto, agrupadas por tipo."

TASKS = {
    # nome            (hf_id,                          config,      split,   conversor)
    "assin2_sts":   ("assin2",                        None,        "train", conv_assin2_sts),
    "assin2_nli":   ("assin2",                        None,        "train", conv_assin2_nli),
    "sick_br_nli":  ("eduagarcia/sick-br",            None,        "train", conv_sick_br),
    "sick_br_sts":  ("eduagarcia/sick-br",            None,        "train", conv_sick_br_sts),
    "faquad":       ("eraldoluis/faquad",             None,        "train", conv_faquad),
    "enem":         ("eduagarcia/enem_challenge",     None,        "train", conv_enem),
    "harem":        ("harem",                         "selective", "train", _conv_ner("ner_tags", NER_HAREM)),
    "lener_br":     ("lener_br",                      None,        "train", _conv_ner("ner_tags", NER_LENER)),
    "macmorpho":    ("mac_morpho",                    None,        "train", conv_macmorpho),
    # split 'test' (2010 ex.); o 'train' do fewshot tem só 75 exemplos.
    "tweetsentbr":  ("eduagarcia/tweetsentbr_fewshot", None,       "test",  conv_tweetsentbr),
    "hatebr":       ("ruanchaves/hatebr",             None,        "train", conv_hatebr),
    "pira":         ("paulopirozelli/pira",           None,        "train", conv_pira),
}

# Conjunto padrão (curado): tarefas mais robustas de conversão automática.
DEFAULT_TASKS = ["assin2_sts", "assin2_nli", "sick_br_nli", "faquad", "enem",
                 "harem", "lener_br", "tweetsentbr", "hatebr", "pira"]


def parse_overrides(items):
    """--source nome=hf_id[:config] -> {nome: (hf_id, config)}."""
    ov = {}
    for it in items or []:
        if "=" not in it:
            continue
        name, rhs = it.split("=", 1)
        hf, _, cfg = rhs.partition(":")
        ov[name.strip()] = (hf.strip(), (cfg.strip() or None))
    return ov


def main() -> int:
    ap = argparse.ArgumentParser(description="manacá-jaster: converte tarefas de NLP PT-BR para SFT (Alpaca)")
    ap.add_argument("--tasks", default="default",
                    help="'all', 'default' ou lista por vírgula de: " + ",".join(TASKS))
    ap.add_argument("--out", default="data/sft/jaster", help="diretório de saída")
    ap.add_argument("--max_per_task", type=int, default=0, help="teto de exemplos por tarefa (0 = sem teto)")
    ap.add_argument("--max_chars", type=int, default=0,
                    help="descarta exemplos cujo prompt (instrução+entrada) passa deste nº de caracteres (0 = off)")
    ap.add_argument("--source", nargs="*", default=None,
                    help="override de dataset: nome=hf_id[:config] (ex.: harem=arubenruben/HAREM-Default)")
    ap.add_argument("--split", default=None, help="força um split para todas as tarefas (ex.: train)")
    ap.add_argument("--shuffle", action="store_true", help="embaralha o mix combinado (semente fixa)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_remote_code", action="store_true",
                    help="não permitir código customizado dos datasets (faquad/harem/lener_br/hatebr precisam)")
    args = ap.parse_args()
    trust = not args.no_remote_code

    try:
        from datasets import load_dataset
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-corpus/manaca-posttrain.")
        return 1

    if args.tasks.strip() == "all":
        selected = list(TASKS)
    elif args.tasks.strip() == "default":
        selected = list(DEFAULT_TASKS)
    else:
        selected = [t.strip() for t in args.tasks.split(",") if t.strip()]

    overrides = parse_overrides(args.source)
    os.makedirs(args.out, exist_ok=True)
    combined_path = os.path.join(args.out, "manaca_jaster.jsonl")
    counts, sha = {}, {}

    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    with open(combined_path, "w", encoding="utf-8") as fout:
        for name in selected:
            if name not in TASKS:
                print(f"[aviso] tarefa desconhecida: {name} (ignorada)"); continue
            hf_id, cfg, split, conv = TASKS[name]
            if name in overrides:
                hf_id, cfg = overrides[name]
            split = args.split or split
            try:
                print(f"[jaster] {name}: baixando {hf_id}" + (f" ({cfg})" if cfg else "") + f" [{split}]...")
                ds = (load_dataset(hf_id, cfg, split=split, trust_remote_code=trust) if cfg
                      else load_dataset(hf_id, split=split, trust_remote_code=trust))
            except Exception as e:  # ID/config/split errado, sem rede, gated... -> pula
                print(f"[aviso] {name}: falha ao carregar {hf_id} ({type(e).__name__}: {e}). PULADA.")
                counts[name] = 0
                continue
            per_path = os.path.join(args.out, f"{name}.jsonl")
            n = 0
            try:
                with open(per_path, "w", encoding="utf-8") as fp:
                    for row in conv(ds):
                        if args.max_per_task and n >= args.max_per_task:
                            break
                        if not row.get("instruction") or not row.get("output"):
                            continue
                        row.setdefault("input", "")
                        if args.max_chars and (len(row["instruction"]) + len(row["input"])) > args.max_chars:
                            continue  # descarta prompt longo (evita corte do marcador no max_seq)
                        row["task"] = name
                        line = json.dumps(row, ensure_ascii=False)
                        fp.write(line + "\n"); fout.write(line + "\n"); n += 1
            except Exception as e:
                print(f"[aviso] {name}: erro na conversão ({type(e).__name__}: {e}). Parcial: {n}.")
            counts[name] = n
            if n:
                sha[name] = sha256(per_path)
            print(f"[jaster] {name}: {n:,} exemplos -> {per_path}")

    if args.shuffle:
        with open(combined_path, encoding="utf-8") as fp:
            lines = fp.readlines()
        random.Random(args.seed).shuffle(lines)
        with open(combined_path, "w", encoding="utf-8") as fp:
            fp.writelines(lines)

    total = sum(counts.values())
    if total:
        sha["manaca_jaster.jsonl"] = sha256(combined_path)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump({"dataset": "manaca-jaster", "tasks": counts, "total": total,
                   "shuffled": args.shuffle, "seed": args.seed, "sha256": sha},
                  fp, ensure_ascii=False, indent=2)
    print(f"[jaster] TOTAL: {total:,} exemplos -> {combined_path}")
    if total == 0:
        print("[jaster] AVISO: nenhum exemplo gerado (verifique rede/IDs; use --source para overrides).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
