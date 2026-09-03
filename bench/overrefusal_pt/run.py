#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manacá-1B — Sonda de OVER-REFUSAL (recusa excessiva) em PT-BR
============================================================
Mede a taxa de recusa do instruct em pedidos BENIGNOS. É o contrapeso do DPO de
segurança: o DPO faz o modelo recusar pedidos nocivos (bom), mas o modo de falha
clássico é ele passar a recusar também pedidos legítimos (over-refusal). Esta
sonda torna esse risco MENSURÁVEL, então a decisão "só fica com o DPO se não
regredir" fica objetiva.

Dois grupos de prompts (bench/overrefusal_pt/prompts.jsonl):
  * benigno            — pedidos comuns (receita, código, e-mail, tradução).
  * sensivel_legitimo  — pedidos que PARECEM perigosos mas são legítimos
                         (proteger o wifi, denunciar um golpe, buscar ajuda
                         emocional). É aqui que a recusa excessiva mais aparece.

Detecção de recusa: reaproveita `eh_recusa` do dpo/gen_dpo_pairs.py (mesmos
marcadores que definem "obediência" no modo safety), para o número ser
consistente com a geração de pares. É uma heurística: as respostas cruas ficam
salvas em answers/<label>.jsonl para conferência manual.

Uso (na imagem manaca-lmeval; veja scripts/eval/run_overrefusal_pt.sh):
    python bench/overrefusal_pt/run.py --model /m --label manaca-instruct-v2

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "dpo"))
from gen_dpo_pairs import montar_prompt, eh_recusa, carregar_modelo  # noqa: E402


def gerar_greedy(tok, model, device, instr, max_new_tokens):
    import torch
    enc = tok(montar_prompt(instr), return_tensors="pt").to(device)
    mi = {"input_ids": enc["input_ids"]}
    if "attention_mask" in enc:
        mi["attention_mask"] = enc["attention_mask"]
    with torch.inference_mode():
        out = model.generate(**mi, do_sample=False, num_beams=1,
                             max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    ini = mi["input_ids"].shape[1]
    return tok.decode(out[0][ini:], skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Sonda de over-refusal (recusa excessiva) PT-BR")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--label", required=True, help="rotulo do modelo (nome do arquivo de saida)")
    ap.add_argument("--prompts", default=os.path.join(REPO, "bench/overrefusal_pt/prompts.jsonl"))
    ap.add_argument("--out_dir", default=os.path.join(REPO, "bench/overrefusal_pt/answers"))
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    with open(args.prompts, encoding="utf-8") as fp:
        prompts = [json.loads(l) for l in fp if l.strip()]

    tok, model, device = carregar_modelo(args.model, args.tokenizer, args.attn, args.dtype)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.label}.jsonl")

    por_tipo = {}   # tipo -> [total, recusas]
    total = recusas = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in prompts:
            resp = gerar_greedy(tok, model, device, p["prompt"], args.max_new_tokens)
            rec = eh_recusa(resp)
            tp = p.get("tipo", "benigno")
            por_tipo.setdefault(tp, [0, 0])
            por_tipo[tp][0] += 1
            por_tipo[tp][1] += int(rec)
            total += 1
            recusas += int(rec)
            out.write(json.dumps({"id": p["id"], "tipo": tp, "prompt": p["prompt"],
                                  "resposta": resp, "recusou": rec},
                                 ensure_ascii=False) + "\n")
            marca = "RECUSOU" if rec else "ok"
            print(f"  [{args.label}] {p['id']:<8} {tp:<18} {marca}")

    print()
    print(f"[over-refusal] modelo={args.label}")
    for tp, (t, r) in sorted(por_tipo.items()):
        print(f"  {tp:<18} recusa {r}/{t} = {100.0*r/t:5.1f}%  (quanto menor, melhor)")
    print(f"  {'GERAL':<18} recusa {recusas}/{total} = {100.0*recusas/total:5.1f}%")
    print(f"[over-refusal] respostas salvas em {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
