#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B — Junta as saidas do lm-eval de uma bateria PT-BR numa tabela
=======================================================================
Generico: le todos os results*.json sob --lm-eval-dir e monta uma tabela
modelo x tarefa com a metrica primaria (acc; fallback acc_norm, depois f1),
com erro padrao. Serve para o Open PT-LLM Leaderboard (enem_pt, bluex_pt,
oab_pt, assin2_rte_pt, faquad_nli_pt, hatebr_pt, hatespeech_pt, ...).

O id do modelo vem do rotulo da PASTA (1o segmento sob --lm-eval-dir), igual ao
merge_pt_benchmarks; quando ha re-run, o mais novo vence (campo `date`).

Uso:
    python3 scripts/eval/merge_ptbench.py --lm-eval-dir docs/evaluation/lmeval_ptbench \\
        --out docs/evaluation/benchmarks-ptleaderboard

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

METRICAS_PRIORIDADE = ["acc", "acc_norm", "f1", "exact_match", "pearson"]


def metrica_primaria(res: dict):
    vals = {}
    for k, v in res.items():
        if not isinstance(v, (int, float)):
            continue
        base = k.split(",")[0]
        vals[base] = float(v)
    for p in METRICAS_PRIORIDADE:
        if p in vals:
            se = vals.get(p + "_stderr")
            return p, vals[p], se
    return None, None, None


def carregar(dirpath: str):
    registros = []  # (date, model_id, task, (val, se))
    for caminho in glob.glob(os.path.join(dirpath, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(caminho, encoding="utf-8"))
        except Exception:
            continue
        if "results" not in d or not isinstance(d["results"], dict):
            continue
        # Usa sempre o rotulo da PASTA (1o segmento sob --lm-eval-dir), que e o
        # LABEL que demos a cada modelo (uniforme p/ locais e do HF).
        seg = os.path.relpath(caminho, dirpath).split(os.sep)
        if seg and seg[0] not in (".", "", os.pardir):
            mid = seg[0]
        else:
            mid = d.get("model_name") or "modelo"
        try:
            dt = float(d.get("date"))
        except (TypeError, ValueError):
            dt = os.path.getmtime(caminho)
        for task, res in d["results"].items():
            _, val, se = metrica_primaria(res)
            if val is None:
                continue
            v = 100.0 * val if val <= 1.0 else val
            s = (100.0 * se) if (se is not None and se <= 1.0) else se
            registros.append((dt, mid, task, (v, s)))
    registros.sort(key=lambda r: r[0])
    dados = {}
    for _, mid, task, cell in registros:
        dados.setdefault(mid, {})[task] = cell
    return dados


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-eval-dir", required=True)
    ap.add_argument("--out", default="docs/evaluation/benchmarks-ptleaderboard")
    ap.add_argument("--ordem", default=None,
                    help="ordem opcional das tarefas (colunas), separadas por virgula")
    a = ap.parse_args()

    dados = carregar(a.lm_eval_dir)
    if not dados:
        print("[merge] nenhum resultado encontrado em", a.lm_eval_dir)
        return 1

    tarefas = sorted({t for m in dados.values() for t in m})
    if a.ordem:
        pref = [t.strip() for t in a.ordem.split(",") if t.strip()]
        tarefas = [t for t in pref if t in tarefas] + [t for t in tarefas if t not in pref]

    def fmt(cell):
        if not cell or cell[0] is None:
            return "-"
        v, s = cell
        return f"{v:.2f} ±{s:.2f}" if isinstance(s, (int, float)) else f"{v:.2f}"

    linhas = ["Valores: metrica primaria (acc) em % ± erro padrao.",
              "",
              "| Modelo | " + " | ".join(tarefas) + " |",
              "|" + "---|" * (len(tarefas) + 1)]
    for mid in sorted(dados):
        estrela = "**" if "manaca" in mid.lower() else ""
        cels = " | ".join(fmt(dados[mid].get(t)) for t in tarefas)
        linhas.append(f"| {estrela}{mid}{estrela} | {cels} |")
    md = "\n".join(linhas) + "\n"

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".md", "w", encoding="utf-8").write(md)
    json.dump(dados, open(a.out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(md)
    print("salvo:", a.out + ".md", "e", a.out + ".json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
