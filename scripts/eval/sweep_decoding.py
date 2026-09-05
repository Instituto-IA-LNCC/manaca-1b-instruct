#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B-instruct - Varredura de configuracao de decoding no IFEval-PT
=======================================================================
Mede, de forma automatica e objetiva, qual configuracao de geracao
(temperatura, top_p, repetition_penalty, greedy) faz o instruct SEGUIR MELHOR as
instrucoes verificaveis do IFEval-PT. Gera as respostas do modelo para cada config
e pontua com os proprios checadores do repo (bench/ifeval_pt/checkers.py), sem juiz.

Metrica: instr-loose (% de instrucoes individuais que passam, com normalizacao).

Uso (dentro do container de eval, com o repo montado em /work):
    MANACA_MODEL=/m python scripts/eval/sweep_decoding.py
    # opcional: MANACA_SEED=0  MANACA_MAXTOK=512

Nota honesta: sao 50 instruicoes (n pequeno) -> diferencas de 1-2 pontos sao ruido.
Escolha a config claramente melhor e confirme o numero final num conjunto held-out
(senao e selecao in-sample). Para paper/leaderboard, use greedy (reprodutivel).

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IFEVAL = os.path.join(REPO, "bench", "ifeval_pt")
sys.path.insert(0, IFEVAL)

PREAMBULO = ("Abaixo está uma instrução que descreve uma tarefa. "
             "Escreva uma resposta que atenda adequadamente ao pedido.")

CONFIGS = {
    "greedy":            dict(do_sample=False),
    "greedy+rep1.15":    dict(do_sample=False, repetition_penalty=1.15),
    "greedy+rep1.30":    dict(do_sample=False, repetition_penalty=1.30),
    "t0.3/p0.9":         dict(do_sample=True, temperature=0.3, top_p=0.9),
    "t0.7/p0.9":         dict(do_sample=True, temperature=0.7, top_p=0.9),
    "t0.7/p0.9+rep1.15": dict(do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.15),
    "t1.0/p0.95":        dict(do_sample=True, temperature=1.0, top_p=0.95),
}


def main() -> int:
    import torch
    import checkers  # de bench/ifeval_pt
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.environ.get("MANACA_MODEL", "/m")
    seed = int(os.environ.get("MANACA_SEED", "0"))
    max_tok = int(os.environ.get("MANACA_MAXTOK", "512"))

    prompts_path = os.path.join(IFEVAL, "prompts.jsonl")
    specs = {p["id"]: p for p in
             (json.loads(l) for l in open(prompts_path, encoding="utf-8") if l.strip())}
    n_inst = sum(len(p["instructions"]) for p in specs.values())
    print(f"[sweep] modelo={model_path}  prompts={len(specs)}  instrucoes={n_inst}  "
          f"seed={seed}  max_new_tokens={max_tok}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()

    def gen(q, cfg):
        text = f"{PREAMBULO}\n\n### Instrução:\n{q}\n\n### Resposta:\n"
        ids = tok(text, return_tensors="pt")
        ids.pop("token_type_ids", None)  # este modelo nao usa; generate() rejeitaria
        ids = ids.to(model.device)
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=max_tok,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id, **cfg)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def pergunta(p):
        # IFEval-PT guarda o enunciado em `turns` (lista, 1 turno); alguns dumps usam `question`.
        if p.get("turns"):
            return p["turns"][0]
        return p.get("question", "")

    def instr_loose(cfg):
        ok = tot = 0
        for p in specs.values():
            r = gen(pergunta(p), cfg)
            for inst in p["instructions"]:
                tot += 1
                ok += int(checkers.checar_instrucao(r, inst, loose=True))
        return 100.0 * ok / tot

    resultados = {}
    for nome, cfg in CONFIGS.items():
        torch.manual_seed(seed)
        score = instr_loose(cfg)
        resultados[nome] = score
        print(f"  {nome:20s} instr-loose = {score:5.1f}%", flush=True)

    melhor = max(resultados, key=resultados.get)
    print(f"\n[sweep] melhor: {melhor} ({resultados[melhor]:.1f}%)  "
          f"-- lembre: n={n_inst}, diferencas de 1-2 pontos sao ruido; "
          f"confirme o final num held-out.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
