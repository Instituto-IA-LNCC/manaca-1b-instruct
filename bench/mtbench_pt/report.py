#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MT-Bench-PT — Relatório e comparação
====================================
Agrega os arquivos julgados (saída do `judge.py`) por modelo: média da nota por
categoria e média geral, com erro padrao (SE = desvio/sqrt(n)). Se receber dois
ou mais modelos, imprime a tabela comparativa e o delta v2 - v1.

Uso:
    python3 bench/mtbench_pt/report.py \\
        bench/mtbench_pt/judged/manaca-instruct-v1.jsonl \\
        bench/mtbench_pt/judged/manaca-instruct-v2.jsonl \\
        --out bench/mtbench_pt/mtbench-pt

Escreve <out>.md e <out>.json. Só biblioteca padrão.

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

ORDEM_CAT = ["escrita", "interpretacao", "raciocinio", "matematica", "codigo",
             "extracao", "ciencias", "humanas", "seguranca"]


def carregar(caminho: str):
    with open(caminho, encoding="utf-8") as fp:
        recs = [json.loads(l) for l in fp if l.strip()]
    label = recs[0].get("model") if recs else os.path.basename(caminho)
    return label, recs


def media_se(valores):
    v = [x for x in valores if isinstance(x, (int, float))]
    if not v:
        return None, None, 0
    m = sum(v) / len(v)
    if len(v) > 1:
        var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
        se = math.sqrt(var / len(v))
    else:
        se = 0.0
    return m, se, len(v)


def agregar(recs):
    por_cat = defaultdict(list)
    todas = []
    parse_err = 0
    for r in recs:
        nota = r.get("nota")
        if nota is None:
            parse_err += 1
            continue
        por_cat[r.get("category", "?")].append(nota)
        todas.append(nota)
    resumo = {}
    for cat, vs in por_cat.items():
        m, se, n = media_se(vs)
        resumo[cat] = {"media": m, "se": se, "n": n}
    m, se, n = media_se(todas)
    resumo["GERAL"] = {"media": m, "se": se, "n": n}
    resumo["_parse_err"] = parse_err
    return resumo


def fmt(cell):
    if not cell or cell.get("media") is None:
        return "-"
    return f"{cell['media']:.2f} ±{cell['se']:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="MT-Bench-PT: relatório")
    ap.add_argument("judged", nargs="+", help="arquivos JSONL julgados (um por modelo)")
    ap.add_argument("--out", default="bench/mtbench_pt/mtbench-pt")
    a = ap.parse_args()

    modelos = []  # (label, resumo)
    for caminho in a.judged:
        label, recs = carregar(caminho)
        modelos.append((label, agregar(recs)))
        print(f"[report] {label}: {len(recs)} respostas, "
              f"parse_err={agregar(recs)['_parse_err']}")

    cats = [c for c in ORDEM_CAT if any(c in r for _, r in modelos)]
    cats += sorted({c for _, r in modelos for c in r
                    if c not in ORDEM_CAT and c not in ("GERAL", "_parse_err")})

    # Tabela markdown
    cab = "| Categoria | " + " | ".join(lbl for lbl, _ in modelos) + " |"
    sep = "|" + "---|" * (len(modelos) + 1)
    linhas = ["Notas MT-Bench-PT (media de 1 a 10 ± erro padrao). Juiz: LLM-as-a-Judge.",
              "", cab, sep]
    for cat in cats:
        cels = " | ".join(fmt(r.get(cat)) for _, r in modelos)
        linhas.append(f"| {cat} | {cels} |")
    geral = " | ".join(fmt(r.get("GERAL")) for _, r in modelos)
    linhas.append(f"| **GERAL** | {geral} |")

    # Delta se houver exatamente 2 modelos (v1 -> v2)
    if len(modelos) == 2:
        (l1, r1), (l2, r2) = modelos
        linhas += ["", f"Delta ({l2} - {l1}) por categoria:", ""]
        linhas += ["| Categoria | Delta |", "|---|---|"]
        for cat in cats + ["GERAL"]:
            c1, c2 = r1.get(cat), r2.get(cat)
            if c1 and c2 and c1.get("media") is not None and c2.get("media") is not None:
                d = c2["media"] - c1["media"]
                linhas.append(f"| {'**'+cat+'**' if cat=='GERAL' else cat} | {d:+.2f} |")

    md = "\n".join(linhas) + "\n"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".md", "w", encoding="utf-8").write(md)
    json.dump({lbl: r for lbl, r in modelos}, open(a.out + ".json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(md)
    print("salvo:", a.out + ".md", "e", a.out + ".json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
