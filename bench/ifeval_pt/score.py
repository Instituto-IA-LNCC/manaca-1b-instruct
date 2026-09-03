#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFEval-PT — Pontuação e comparação (sem juiz)
=============================================
Le as respostas geradas (mesmo formato do gen_answers.py, um arquivo por modelo)
e o `prompts.jsonl` (specs das instrucoes), aplica os checadores e reporta, no
padrao do IFEval, quatro metricas por modelo:

  * prompt-strict : % de prompts em que TODAS as instrucoes passam (resposta crua)
  * prompt-loose  : idem, com normalizacao (variantes)
  * instr-strict  : % de instrucoes (individuais) que passam (resposta crua)
  * instr-loose   : idem, com normalizacao

Tambem imprime a acuracia loose por TIPO de instrucao (onde os modelos falham).

Uso:
    python3 bench/ifeval_pt/score.py \\
        bench/ifeval_pt/answers/manaca-instruct-v1.jsonl \\
        bench/ifeval_pt/answers/manaca-instruct-v2.jsonl \\
        ... \\
        --prompts bench/ifeval_pt/prompts.jsonl --out bench/ifeval_pt/ifeval-pt

Só biblioteca padrão.

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkers  # noqa: E402


def _carregar_jsonl(caminho):
    with open(caminho, encoding="utf-8") as fp:
        return [json.loads(l) for l in fp if l.strip()]


def _prop_se(acertos, total):
    if not total:
        return None, None
    p = acertos / total
    se = math.sqrt(p * (1 - p) / total)
    return 100.0 * p, 100.0 * se


def avaliar_modelo(respostas, specs):
    """Devolve (metricas, por_tipo) para um modelo.
    specs: {id: [instrucoes]}. respostas: lista de {id, answer, model}."""
    resp_por_id = {r["id"]: r.get("answer", "") for r in respostas}
    n_prompts = ok_prompt_strict = ok_prompt_loose = 0
    n_inst = ok_inst_strict = ok_inst_loose = 0
    tipo_tot = defaultdict(int); tipo_ok_loose = defaultdict(int); tipo_ok_strict = defaultdict(int)

    for pid, insts in specs.items():
        if pid not in resp_por_id:
            continue
        resp = resp_por_id[pid]
        n_prompts += 1
        todas_strict = todas_loose = True
        for inst in insts:
            s = checkers.checar_instrucao(resp, inst, loose=False)
            l = checkers.checar_instrucao(resp, inst, loose=True)
            n_inst += 1
            ok_inst_strict += int(s); ok_inst_loose += int(l)
            tipo_tot[inst["type"]] += 1
            tipo_ok_strict[inst["type"]] += int(s); tipo_ok_loose[inst["type"]] += int(l)
            todas_strict = todas_strict and s
            todas_loose = todas_loose and l
        ok_prompt_strict += int(todas_strict); ok_prompt_loose += int(todas_loose)

    ps, ps_se = _prop_se(ok_prompt_strict, n_prompts)
    pl, pl_se = _prop_se(ok_prompt_loose, n_prompts)
    is_, is_se = _prop_se(ok_inst_strict, n_inst)
    il, il_se = _prop_se(ok_inst_loose, n_inst)
    metr = {"prompt_strict": (ps, ps_se), "prompt_loose": (pl, pl_se),
            "instr_strict": (is_, is_se), "instr_loose": (il, il_se),
            "n_prompts": n_prompts, "n_inst": n_inst}
    por_tipo = {t: (100.0 * tipo_ok_loose[t] / tipo_tot[t],
                    100.0 * tipo_ok_strict[t] / tipo_tot[t], tipo_tot[t])
                for t in sorted(tipo_tot)}
    return metr, por_tipo


def _fmt(cell):
    if not cell or cell[0] is None:
        return "-"
    return f"{cell[0]:.1f} ±{cell[1]:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="IFEval-PT: pontuacao")
    ap.add_argument("answers", nargs="+", help="arquivos JSONL de respostas (um por modelo)")
    ap.add_argument("--prompts", default="bench/ifeval_pt/prompts.jsonl")
    ap.add_argument("--out", default="bench/ifeval_pt/ifeval-pt")
    a = ap.parse_args()

    specs = {p["id"]: p["instructions"] for p in _carregar_jsonl(a.prompts)}
    print(f"[ifeval] {len(specs)} prompts, "
          f"{sum(len(v) for v in specs.values())} instrucoes verificaveis")

    modelos = []
    por_tipo_por_modelo = {}
    for caminho in a.answers:
        recs = _carregar_jsonl(caminho)
        label = recs[0].get("model") if recs else os.path.basename(caminho)
        metr, por_tipo = avaliar_modelo(recs, specs)
        modelos.append((label, metr))
        por_tipo_por_modelo[label] = por_tipo
        print(f"[ifeval] {label}: prompt-strict={_fmt(metr['prompt_strict'])}  "
              f"instr-loose={_fmt(metr['instr_loose'])}")

    # Tabela principal
    linhas = ["Acuracia IFEval-PT (%) ± erro padrao. strict=resposta crua; loose=normalizada.",
              "",
              "| Modelo | Prompt strict | Prompt loose | Instr strict | Instr loose |",
              "|---|---|---|---|---|"]
    for label, m in modelos:
        linhas.append(f"| {label} | {_fmt(m['prompt_strict'])} | {_fmt(m['prompt_loose'])} | "
                      f"{_fmt(m['instr_strict'])} | {_fmt(m['instr_loose'])} |")

    # Tabela por tipo (instr-loose por tipo, um modelo por coluna)
    tipos = sorted({t for pt in por_tipo_por_modelo.values() for t in pt})
    linhas += ["", "Instr-loose (%) por tipo de instrucao:", "",
               "| Tipo | " + " | ".join(l for l, _ in modelos) + " |",
               "|" + "---|" * (len(modelos) + 1)]
    for t in tipos:
        cels = []
        for label, _ in modelos:
            pt = por_tipo_por_modelo[label].get(t)
            cels.append(f"{pt[0]:.0f}" if pt else "-")
        linhas.append(f"| {t} | " + " | ".join(cels) + " |")

    md = "\n".join(linhas) + "\n"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".md", "w", encoding="utf-8").write(md)
    json.dump({"modelos": {l: m for l, m in modelos},
               "por_tipo": por_tipo_por_modelo},
              open(a.out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(md)
    print("salvo:", a.out + ".md", "e", a.out + ".json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
