#!/usr/bin/env python3
"""
Manacá-1B — Fase 3a · Preparo dos dados de SFT | mix de instrução em PT-BR
=========================================================================
PT
--
Monta o mix de instrução/conversa em PT-BR análogo ao do llm-jp-sft (Jaster +
Dolly-ja + OASST-ja), normalizado para o esquema Alpaca `{instruction, input,
output}` que o `sft/train.py` consome.

Fontes (todas com carregamento tolerante: uma fora do ar é PULADA, não derruba
o mix):
  - alpaca        : dominguesm/alpaca-data-pt-br   (Alpaca-PT ~52k; análogo Dolly-ja)
  - aya           : CohereForAI/aya_dataset (pt)   (instruções humanas nativas + tradução)
  - oasst         : OpenAssistant/oasst1 (lang=pt) (conversa; análogo OASST-ja)
  - translation   : Helsinki-NLP/opus-100 (en-pt)  (tradução PT<->EN, ambos sentidos)  [v2]
  - summarization : csebuetnlp/xlsum (portuguese)  (sumarização)                        [v2]

A metade "tarefas de NLP" (o manacá-jaster) é gerada por `sft/jaster/build_jaster.py`
e entra aqui via `--extra`.

VERSIONAMENTO (v1 vs v2): controle por `--out`. A v1 usou `--sources alpaca,oasst
--out data/sft`. Para a v2, use `--sources alpaca,aya,oasst,translation,summarization
--out data/sft_v2` (escreve em outro diretório, sem sobrescrever a v1). O
`--max_chars` descarta exemplos cujo prompt (instrução+entrada) é longo demais
(evita o corte do marcador de resposta no `max_seq_length`).

EN
--
Builds the PT-BR instruction/conversation mix (Alpaca schema). Each source loads
tolerantly (skipped with a warning on failure). Use `--out` to keep v1 and v2
side by side (different directories, nothing overwritten).

Uso | Usage:
    python sft/prepare_data.py --sources alpaca,aya,oasst,translation,summarization \\
        --out data/sft_v2 --shuffle --dedup --max_chars 5000 \\
        --extra data/sft_v2/jaster/manaca_jaster.jsonl

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys


def _s(x):
    return ("" if x is None else str(x)).strip()


# ── Normalizadores por fonte (geram {instruction, input, output}) ─────────────
def norm_alpaca(ds):
    """Alpaca-PT (instruction/input/output), ex.: dominguesm/alpaca-data-pt-br."""
    for ex in ds:
        instr, out = _s(ex.get("instruction")), _s(ex.get("output"))
        if instr and out:
            yield {"instruction": instr, "input": _s(ex.get("input")), "output": out}


def norm_aya(ds):
    """CohereForAI/aya_dataset: filtra português (language/ language_code)."""
    for ex in ds:
        lang = _s(ex.get("language")).lower()
        code = _s(ex.get("language_code")).lower()
        if not (code.startswith("pt") or "portug" in lang):
            continue
        instr, out = _s(ex.get("inputs")), _s(ex.get("targets"))
        if instr and out:
            yield {"instruction": instr, "input": "", "output": out}


def norm_oasst(ds):
    """OpenAssistant/oasst1: pares prompter->assistant em pt (melhor/única resposta).
    Filtros relaxados (v2): lang começando com 'pt', melhor resposta (rank 0 ou
    None), pai = prompter — sem exigir lang do pai (corrige o 0 exemplos da v1)."""
    def is_pt(x):
        return _s(x).lower().startswith("pt")
    msgs = {m["message_id"]: m for m in ds}
    for m in ds:
        if m.get("role") != "assistant" or m.get("deleted"):
            continue
        if not is_pt(m.get("lang")):
            continue
        rank = m.get("rank")
        if rank is not None and rank != 0:      # melhor resposta do nó (ou única)
            continue
        parent = msgs.get(m.get("parent_id"))
        if not parent or parent.get("role") != "prompter":
            continue
        instr, out = _s(parent.get("text")), _s(m.get("text"))
        if instr and out:
            yield {"instruction": instr, "input": "", "output": out}


def norm_opus(ds):
    """Helsinki-NLP/opus-100 (en-pt): tradução nos dois sentidos."""
    for ex in ds:
        t = ex.get("translation") or {}
        en, pt = _s(t.get("en")), _s(t.get("pt"))
        if not en or not pt:
            continue
        yield {"instruction": "Traduza o texto a seguir para o inglês.", "input": pt, "output": en}
        yield {"instruction": "Traduza o texto a seguir para o português.", "input": en, "output": pt}


def norm_xlsum(ds):
    """csebuetnlp/xlsum (portuguese): sumarização (text -> summary)."""
    for ex in ds:
        text, summ = _s(ex.get("text")), _s(ex.get("summary"))
        if not text or not summ:
            continue
        yield {"instruction": "Resuma o texto a seguir em poucas frases.", "input": text, "output": summ}


# nome: (hf_id, config, split, normalizador)
SOURCES = {
    "alpaca":        ("dominguesm/alpaca-data-pt-br", None,         "train",          norm_alpaca),
    "aya":           ("CohereForAI/aya_dataset",      None,         "train",          norm_aya),
    "oasst":         ("OpenAssistant/oasst1",         None,         "train",          norm_oasst),
    "translation":   ("Helsinki-NLP/opus-100",        "en-pt",      "train[:20000]",  norm_opus),
    "summarization": ("csebuetnlp/xlsum",             "portuguese", "train[:10000]",  norm_xlsum),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Preparo dos dados de SFT do Manacá (mix PT-BR)")
    ap.add_argument("--sources", default="alpaca,aya,oasst,translation,summarization",
                    help="subconjunto de: " + ",".join(SOURCES))
    ap.add_argument("--out", default="data/sft", help="diretório de saída (use data/sft_v2 para a v2)")
    ap.add_argument("--max_per_source", type=int, default=0, help="teto de exemplos por fonte (0 = sem teto)")
    ap.add_argument("--max_chars", type=int, default=0,
                    help="descarta exemplos cujo prompt (instrução+entrada) passa deste nº de caracteres (0 = off)")
    ap.add_argument("--dedup", action="store_true", help="remove (instruction,input,output) duplicados")
    ap.add_argument("--shuffle", action="store_true", help="embaralha o mix (semente fixa)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra", nargs="*", default=None, help=".jsonl Alpaca extras (ex.: manacá-jaster)")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-corpus/manaca-posttrain.")
        return 1

    os.makedirs(args.out, exist_ok=True)
    combined_path = os.path.join(args.out, "manaca_sft.jsonl")
    seen = set()
    counts = {}

    def too_long(r):
        return args.max_chars and (len(r.get("instruction", "")) + len(r.get("input", ""))) > args.max_chars

    def emit(fout, rows, tag):
        n = 0
        per_path = os.path.join(args.out, f"{tag}.jsonl")
        with open(per_path, "w", encoding="utf-8") as fp:
            for r in rows:
                if args.max_per_source and n >= args.max_per_source:
                    break
                if too_long(r):
                    continue
                if args.dedup:
                    key = hash((r["instruction"], r.get("input", ""), r["output"]))
                    if key in seen:
                        continue
                    seen.add(key)
                r.setdefault("input", "")
                line = json.dumps(r, ensure_ascii=False)
                fp.write(line + "\n"); fout.write(line + "\n"); n += 1
        counts[tag] = n
        print(f"[data] {tag}: {n:,} exemplos -> {per_path}")

    with open(combined_path, "w", encoding="utf-8") as fout:
        for tag in [s.strip() for s in args.sources.split(",") if s.strip()]:
            if tag not in SOURCES:
                print(f"[aviso] fonte desconhecida: {tag} (ignorada)"); continue
            hf_id, cfg, split, norm = SOURCES[tag]
            print(f"[data] baixando {hf_id}" + (f" ({cfg})" if cfg else "") + f" [{split}]...")
            try:  # uma fonte fora do ar não derruba o mix inteiro
                ds = (load_dataset(hf_id, cfg, split=split, trust_remote_code=True) if cfg
                      else load_dataset(hf_id, split=split, trust_remote_code=True))
            except Exception as e:
                print(f"[aviso] {tag}: falha ao carregar {hf_id} ({type(e).__name__}: {e}). PULADA.")
                counts[tag] = 0
                continue
            emit(fout, norm(ds), tag)
        if args.extra:
            for path in args.extra:
                if not os.path.exists(path):
                    print(f"[aviso] extra ausente: {path} (ignorado)"); continue
                def _read(p=path):
                    with open(p, encoding="utf-8") as fp:
                        for line in fp:
                            line = line.strip()
                            if line:
                                yield json.loads(line)
                emit(fout, _read(), os.path.splitext(os.path.basename(path))[0])

    if args.shuffle:
        with open(combined_path, encoding="utf-8") as fp:
            lines = fp.readlines()
        random.Random(args.seed).shuffle(lines)
        with open(combined_path, "w", encoding="utf-8") as fp:
            fp.writelines(lines)

    total = sum(counts.values())
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump({"sources": counts, "total": total, "shuffled": args.shuffle,
                   "seed": args.seed, "dedup": args.dedup, "max_chars": args.max_chars,
                   "max_per_source": args.max_per_source}, fp, ensure_ascii=False, indent=2)
    print(f"[data] TOTAL: {total:,} exemplos -> {combined_path}")
    print(f"[data] manifest -> {os.path.join(args.out, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
