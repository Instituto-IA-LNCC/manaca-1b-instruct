#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B — f1-macro (e acc) a partir dos samples do lm-eval
============================================================
As tarefas de classificacao PT-BR sao desbalanceadas, entao a acc favorece a
classe majoritaria. O numero rigoroso (e o do Open PT-LLM Leaderboard) e o
f1-macro. Este script recalcula f1-macro SEM re-rodar o modelo: le os
`samples_*.jsonl` que o lm-eval ja gravou (com o gold e as loglikelihoods por
alternativa) e, para cada exemplo, toma pred = argmax das loglikelihoods.

Identifica modelo/tarefa pelo CAMINHO: <lm-eval-dir>/<modelo>/<tarefa>/.../samples_*.jsonl

Uso:
    python3 scripts/eval/f1_ptbench.py --lm-eval-dir docs/evaluation/lmeval_ptbench \\
        --tasks assin2_rte_pt,faquad_nli_pt,hatebr_pt,hatespeech_pt \\
        --out docs/evaluation/benchmarks-ptleaderboard-f1

So biblioteca padrao.

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

CLASSIF_DEFAULT = ["assin2_rte_pt", "faquad_nli_pt", "hatebr_pt", "hatespeech_pt"]


def _abrir(caminho):
    if caminho.endswith(".gz"):
        return gzip.open(caminho, "rt", encoding="utf-8")
    return open(caminho, encoding="utf-8")


def _ll(r):
    """Extrai a loglikelihood de um item de filtered_resps (varios formatos)."""
    v = r
    while isinstance(v, (list, tuple)) and v:
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("-inf")


def gold_pred(s):
    gold = s.get("target")
    if isinstance(gold, (list, tuple)):
        gold = gold[0] if gold else None
    try:
        gold = int(gold)
    except (TypeError, ValueError):
        gold = None
    resps = s.get("filtered_resps") or s.get("resps")
    pred = None
    if resps:
        lls = [_ll(r) for r in resps]
        if lls:
            pred = max(range(len(lls)), key=lambda i: lls[i])
    return gold, pred


def f1_macro(golds, preds):
    classes = sorted(set(golds) | set(preds))
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in zip(golds, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(golds, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(golds, preds) if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return (sum(f1s) / len(f1s)) if f1s else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-eval-dir", required=True)
    ap.add_argument("--tasks", default=",".join(CLASSIF_DEFAULT),
                    help="tarefas (por caminho) a incluir, separadas por virgula")
    ap.add_argument("--out", default="docs/evaluation/benchmarks-ptleaderboard-f1")
    ap.add_argument("--debug", action="store_true", help="imprime as chaves do 1o sample")
    a = ap.parse_args()
    tarefas_alvo = [t.strip() for t in a.tasks.split(",") if t.strip()]

    # (modelo, tarefa) -> (golds, preds), pegando o samples mais novo por par
    por_par = {}
    novos = {}
    for caminho in glob.glob(os.path.join(a.lm_eval_dir, "**", "samples_*.jsonl*"), recursive=True):
        seg = os.path.relpath(caminho, a.lm_eval_dir).split(os.sep)
        if len(seg) < 2:
            continue
        modelo, tarefa = seg[0], seg[1]
        if tarefa not in tarefas_alvo:
            continue
        mt = os.path.getmtime(caminho)
        if (modelo, tarefa) in novos and novos[(modelo, tarefa)] >= mt:
            continue  # ja temos um mais novo
        golds, preds = [], []
        primeiro = True
        with _abrir(caminho) as fp:
            for linha in fp:
                if not linha.strip():
                    continue
                s = json.loads(linha)
                if a.debug and primeiro:
                    print(f"[debug] {modelo}/{tarefa} chaves:", list(s.keys())); primeiro = False
                g, p = gold_pred(s)
                if g is None or p is None:
                    continue
                golds.append(g); preds.append(p)
        if golds:
            por_par[(modelo, tarefa)] = (golds, preds)
            novos[(modelo, tarefa)] = mt

    if not por_par:
        print("[f1] nenhum samples valido encontrado. Rode com --debug para ver as chaves.")
        return 1

    modelos = sorted({m for m, _ in por_par})
    tarefas = [t for t in tarefas_alvo if any((m, t) in por_par for m in modelos)]

    linhas_md = ["f1-macro (%) por modelo/tarefa (recalculado dos samples do lm-eval).",
                 "acc entre parenteses para referencia.",
                 "",
                 "| Modelo | " + " | ".join(tarefas) + " |",
                 "|" + "---|" * (len(tarefas) + 1)]
    dados = {}
    for m in modelos:
        cels = []
        dados[m] = {}
        for t in tarefas:
            if (m, t) in por_par:
                g, p = por_par[(m, t)]
                f1 = 100.0 * f1_macro(g, p)
                acc = 100.0 * sum(1 for a_, b_ in zip(g, p) if a_ == b_) / len(g)
                dados[m][t] = {"f1_macro": f1, "acc": acc, "n": len(g)}
                cels.append(f"{f1:.2f} ({acc:.1f})")
            else:
                cels.append("-")
        estrela = "**" if "manaca" in m.lower() else ""
        linhas_md.append(f"| {estrela}{m}{estrela} | " + " | ".join(cels) + " |")
    md = "\n".join(linhas_md) + "\n"

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".md", "w", encoding="utf-8").write(md)
    json.dump(dados, open(a.out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(md)
    print("salvo:", a.out + ".md", "e", a.out + ".json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
