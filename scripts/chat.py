#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat interativo simples com um instruct do Manacá (template Alpaca-PT)
=====================================================================
Carrega um modelo instruct e responde perguntas no terminal, usando o MESMO
template do SFT (é o formato que o modelo espera). Single-turn: cada pergunta é
independente (o modelo não foi treinado para multi-turno).

Uso (via scripts/run_chat.sh, dentro do docker):
    ./scripts/run_chat.sh                      # chat interativo (v2 por padrão)
    MODEL=/caminho ./scripts/run_chat.sh       # outro modelo
    ./scripts/run_chat.sh --greedy             # determinístico
    ./scripts/run_chat.sh --prompt "Quem foi Machado de Assis?"   # uma pergunta só

Comandos no chat: escreva a pergunta e Enter. 'sair' (ou Ctrl-D) encerra.

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import sys

PREAMBLE_NOINPUT = (
    "Abaixo está uma instrução que descreve uma tarefa. "
    "Escreva uma resposta que atenda adequadamente ao pedido."
)
INSTRUCTION_TEMPLATE = "\n\n### Instrução:\n"
RESPONSE_TEMPLATE = "\n\n### Resposta:\n"


def montar_prompt(instrucao: str) -> str:
    return f"{PREAMBLE_NOINPUT}{INSTRUCTION_TEMPLATE}{instrucao.strip()}{RESPONSE_TEMPLATE}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Chat com o instruct do Manacá")
    ap.add_argument("--model", default="/m")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--greedy", action="store_true", help="determinístico (sem amostragem)")
    ap.add_argument("--prompt", default=None, help="faz UMA pergunta e sai")
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[ERRO] dependência ausente ({e}). Rode na imagem manaca-lmeval.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[chat] carregando {args.model} em {device} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn).to(device)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    def responder(instr: str) -> str:
        enc = tok(montar_prompt(instr), return_tensors="pt").to(device)
        mi = {"input_ids": enc["input_ids"]}
        if "attention_mask" in enc:
            mi["attention_mask"] = enc["attention_mask"]
        kw = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
                  eos_token_id=tok.eos_token_id)
        if args.greedy:
            kw.update(do_sample=False)
        else:
            kw.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
        with torch.inference_mode():
            out = model.generate(**mi, **kw)
        return tok.decode(out[0][mi["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    if args.prompt:
        print(responder(args.prompt))
        return 0

    print("[chat] pronto. Digite a pergunta e Enter. 'sair' ou Ctrl-D encerra.\n", file=sys.stderr)
    while True:
        try:
            instr = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not instr:
            continue
        if instr.lower() in ("sair", "exit", "quit"):
            break
        print("Manacá:", responder(instr), "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
