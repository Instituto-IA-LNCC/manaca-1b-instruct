#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MT-Bench-PT — Geração das respostas do modelo instruct
======================================================
Gera as respostas de UM modelo para o conjunto `questions.jsonl`, usando o MESMO
template Alpaca-PT do SFT/DPO (o modelo é interrogado do jeito que foi treinado,
como faz o llm-jp-eval em modo instruct). A temperatura segue o padrão do
MT-Bench: 0 (determinístico) para tarefas com resposta certa/errada e 0.7 para
tarefas abertas.

Saída: um JSONL com {id, category, question, answer, model, gen} pronto para o
juiz (`judge.py`).

Uso (na imagem de avaliação, com o modelo montado em /m — veja
scripts/eval/run_mtbench_pt.sh):
    python bench/mtbench_pt/gen_answers.py \\
        --model /m --model_label manaca-instruct-v2 \\
        --questions bench/mtbench_pt/questions.jsonl \\
        --out bench/mtbench_pt/answers/manaca-instruct-v2.jsonl

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Preâmbulos e marcadores IDÊNTICOS ao SFT (sft/train.py) e ao DPO (dpo/train.py).
PREAMBLE_NOINPUT = (
    "Abaixo está uma instrução que descreve uma tarefa. "
    "Escreva uma resposta que atenda adequadamente ao pedido."
)
PREAMBLE_INPUT = (
    "Abaixo está uma instrução que descreve uma tarefa, combinada com uma entrada "
    "que fornece mais contexto. Escreva uma resposta que atenda adequadamente ao pedido."
)
INSTRUCTION_TEMPLATE = "\n\n### Instrução:\n"
INPUT_TEMPLATE = "\n\n### Entrada:\n"
RESPONSE_TEMPLATE = "\n\n### Resposta:\n"

# Temperatura por categoria (padrão MT-Bench: criativo amostra, objetivo é greedy).
TEMP_POR_CATEGORIA = {
    "escrita": 0.7,
    "interpretacao": 0.7,
    "ciencias": 0.7,
    "humanas": 0.7,
    "raciocinio": 0.0,
    "matematica": 0.0,
    "codigo": 0.0,
    "extracao": 0.0,
    "seguranca": 0.0,
}


def montar_prompt(instrucao: str, entrada: str = "") -> str:
    entrada = (entrada or "").strip()
    if entrada:
        return (f"{PREAMBLE_INPUT}{INSTRUCTION_TEMPLATE}{instrucao.strip()}"
                f"{INPUT_TEMPLATE}{entrada}{RESPONSE_TEMPLATE}")
    return f"{PREAMBLE_NOINPUT}{INSTRUCTION_TEMPLATE}{instrucao.strip()}{RESPONSE_TEMPLATE}"


def construir_entrada(tok, instrucao: str, style: str, device):
    """Monta os input_ids do jeito CERTO para cada modelo (justica na comparacao):
    - 'chat'  : usa o chat_template do proprio modelo (apply_chat_template);
    - 'alpaca': usa o template Alpaca-PT do SFT do Manaca;
    - 'auto'  : chat_template se o modelo tiver um; senao Alpaca-PT.
    Devolve (model_inputs, style_usado)."""
    tem_ct = getattr(tok, "chat_template", None)
    usar_chat = style == "chat" or (style == "auto" and tem_ct)
    if usar_chat and not tem_ct:
        raise SystemExit("[ERRO] --prompt_style chat, mas o tokenizador nao tem chat_template.")
    if usar_chat:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": instrucao}],
            add_generation_prompt=True, return_tensors="pt").to(device)
        return {"input_ids": ids}, "chat"
    enc = tok(montar_prompt(instrucao), return_tensors="pt").to(device)
    mi = {"input_ids": enc["input_ids"]}
    if "attention_mask" in enc:
        mi["attention_mask"] = enc["attention_mask"]
    return mi, "alpaca"


def main() -> int:
    ap = argparse.ArgumentParser(description="MT-Bench-PT: geração de respostas")
    ap.add_argument("--model", required=True, help="Caminho local (/m) ou id HF do modelo")
    ap.add_argument("--model_label", default=None, help="Rótulo do modelo na saída (ex.: manaca-instruct-v2)")
    ap.add_argument("--tokenizer", default=None, help="Tokenizador (default: o do próprio modelo)")
    ap.add_argument("--questions", default="bench/mtbench_pt/questions.jsonl")
    ap.add_argument("--out", required=True, help="JSONL de saída com as respostas")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn", default="sdpa", help="flash_attention_2 | sdpa | eager")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prompt_style", default="auto", choices=["auto", "chat", "alpaca"],
                    help="auto: chat_template do modelo se existir, senao Alpaca-PT do SFT")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as e:
        print(f"[ERRO] dependência ausente ({e}). Rode na imagem de avaliação (manaca-lmeval).")
        return 1

    set_seed(args.seed)
    label = args.model_label or os.path.basename(args.model.rstrip("/")) or "modelo"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    print(f"[mtbench] modelo={args.model} rótulo={label} device={device}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, attn_implementation=args.attn).to(device)
    except Exception:
        # fallback sem flash/sdpa se a imagem não suportar
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    with open(args.questions, encoding="utf-8") as fp:
        perguntas = [json.loads(l) for l in fp if l.strip()]
    print(f"[mtbench] {len(perguntas)} perguntas")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for q in perguntas:
            instr = q["turns"][0]
            cat = q.get("category", "")
            temp = TEMP_POR_CATEGORIA.get(cat, 0.0)
            model_inputs, style_usado = construir_entrada(tok, instr, args.prompt_style, device)
            gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                              pad_token_id=tok.pad_token_id,
                              eos_token_id=tok.eos_token_id)
            if temp and temp > 0:
                gen_kwargs.update(do_sample=True, temperature=temp, top_p=args.top_p)
            else:
                gen_kwargs.update(do_sample=False)
            with __import__("torch").inference_mode():
                out_ids = model.generate(**model_inputs, **gen_kwargs)
            texto = tok.decode(out_ids[0][model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            resposta = texto.strip()
            rec = {"id": q["id"], "category": cat, "question": instr,
                   "reference": q.get("reference"), "answer": resposta,
                   "model": label, "gen": {"temperature": temp, "top_p": args.top_p,
                                            "max_new_tokens": args.max_new_tokens,
                                            "prompt_style": style_usado}}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n += 1
            print(f"  [{n:>2}/{len(perguntas)}] {q['id']:<14} (T={temp}) "
                  f"-> {len(resposta)} chars")
    print(f"[mtbench] respostas salvas -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
