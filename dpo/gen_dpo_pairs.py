#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manacá-1B — Gerador de pares de preferência ON-POLICY para o DPO
================================================================
Corrige a falha do nosso 1o DPO (pares off-policy do GigaVerbo): aqui os
candidatos são gerados PELO PRÓPRIO instruct (ex.: manaca-1b-instruct-v2-full) e
depois rotulados. Fiel ao método do LLM-jp (dados por juiz + linha de segurança
tipo AnswerCarefully), adaptado à nossa realidade (on-policy, PT-BR, focado nos
gaps que os benchmarks mediram: segurança e seguir instrução).

Três fluxos (--mode):
  * safety      (regra): pega pedidos nocivos (dpo_seeds/safety_prompts.jsonl),
                 amostra o modelo; se ele OBEDECE, isso vira `rejected` e uma
                 RECUSA boa vira `chosen`. Se o modelo já recusa, pula.
  * instruction (regra): usa os prompts + checkers do IFEval-PT; amostra k, o que
                 PASSA nas restrições vira `chosen`, o que FALHA vira `rejected`.
  * quality     (juiz): prompts abertos (MT-Bench-PT); amostra 2, o juiz LLM
                 (mesma infra do bench, claude-opus-5) escolhe chosen/rejected.

Saída: JSONL {instruction, input, chosen, rejected} pronto para dpo/train.py.
ATENÇÃO: a saída do modo safety contém a obediência nociva do modelo em
`rejected`. NÃO versione (dpo_onpolicy/ está no .gitignore).

Uso (na imagem manaca-lmeval; veja scripts/run_gen_dpo_pairs.sh):
    python dpo/gen_dpo_pairs.py --mode safety      --model /m --out dpo_onpolicy/safety.jsonl
    python dpo/gen_dpo_pairs.py --mode instruction --model /m --out dpo_onpolicy/instruction.jsonl
    python dpo/gen_dpo_pairs.py --mode quality     --model /m --out dpo_onpolicy/quality.jsonl  # precisa ANTHROPIC_API_KEY

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREAMBLE_NOINPUT = (
    "Abaixo está uma instrução que descreve uma tarefa. "
    "Escreva uma resposta que atenda adequadamente ao pedido."
)
INSTRUCTION_TEMPLATE = "\n\n### Instrução:\n"
RESPONSE_TEMPLATE = "\n\n### Resposta:\n"


def montar_prompt(instr: str) -> str:
    return f"{PREAMBLE_NOINPUT}{INSTRUCTION_TEMPLATE}{instr.strip()}{RESPONSE_TEMPLATE}"


# ── Recusas (chosen do modo safety), por categoria ────────────────────────────
REFUSAL_DEFAULT = [
    "Desculpe, mas não posso ajudar com isso. O pedido envolve uma atividade ilegal "
    "ou que pode causar dano a outras pessoas. Se você tiver um objetivo legítimo por "
    "trás disso, posso sugerir um caminho seguro e dentro da lei.",
    "Não vou ajudar com esse pedido, porque ele pode causar dano ou é ilegal. Se quiser, "
    "posso explicar os riscos envolvidos ou indicar alternativas legítimas.",
    "Sinto muito, mas isso é algo com que não posso ajudar. Envolve risco de dano ou "
    "quebra da lei. Posso te ajudar com uma alternativa segura, se você quiser.",
]
REFUSAL_AUTOLESAO = [
    "Sinto muito que você esteja passando por isso, e obrigado por confiar em mim. Não "
    "vou ajudar com isso, mas você não está sozinho. Se estiver em sofrimento, ligue para "
    "o CVV no número 188 (24h, gratuito) ou acesse cvv.org.br. Falar com alguém de "
    "confiança ou um profissional de saúde pode ajudar muito.",
]


def escolher_recusa(categoria: str, rng: random.Random) -> str:
    if categoria == "autolesao":
        return rng.choice(REFUSAL_AUTOLESAO)
    return rng.choice(REFUSAL_DEFAULT)


REFUSAL_MARKERS = [
    "não posso", "nao posso", "não vou", "nao vou", "desculpe", "não é apropriado",
    "nao e apropriado", "não consigo ajudar", "nao consigo ajudar", "isso é ilegal",
    "isso e ilegal", "não devo", "nao devo", "procure ajuda", "cvv",
]


def eh_recusa(texto: str) -> bool:
    t = texto.lower()
    return any(m in t for m in REFUSAL_MARKERS)


# ── Modelo | Model ────────────────────────────────────────────────────────────
def carregar_modelo(model_path, tokenizer_path, attn, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = getattr(torch, dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dt, attn_implementation=attn).to(device)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dt).to(device)
    model.eval()
    return tok, model, device


def amostrar(tok, model, device, instr, n, temperature, top_p, max_new_tokens):
    import torch
    enc = tok(montar_prompt(instr), return_tensors="pt").to(device)
    mi = {"input_ids": enc["input_ids"]}
    if "attention_mask" in enc:
        mi["attention_mask"] = enc["attention_mask"]
    with torch.inference_mode():
        out = model.generate(**mi, do_sample=True, temperature=temperature, top_p=top_p,
                             num_return_sequences=n, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    ini = mi["input_ids"].shape[1]
    return [tok.decode(o[ini:], skip_special_tokens=True).strip() for o in out]


def _carregar_jsonl(caminho):
    with open(caminho, encoding="utf-8") as fp:
        return [json.loads(l) for l in fp if l.strip()]


# ── Fluxos ────────────────────────────────────────────────────────────────────
def modo_safety(args, tok, model, device, rng, out):
    prompts = _carregar_jsonl(args.safety_prompts)
    n = kept = 0
    for p in prompts:
        instr = p["prompt"]; cat = p.get("categoria", "default")
        respostas = amostrar(tok, model, device, instr, args.k, args.temperature,
                             args.top_p, args.max_new_tokens)
        for r in respostas:
            n += 1
            if not r or eh_recusa(r):
                continue  # modelo ja recusou (bom) -> sem sinal util
            chosen = escolher_recusa(cat, rng)
            out.write(json.dumps({"instruction": instr, "input": "",
                                  "chosen": chosen, "rejected": r,
                                  "meta": {"mode": "safety", "categoria": cat}},
                                 ensure_ascii=False) + "\n")
            kept += 1
        print(f"  [safety] {p['id']:<8} {cat:<14} amostras={len(respostas)}")
    print(f"[safety] pares gerados: {kept} (de {n} amostras)")
    return kept


def modo_instruction(args, tok, model, device, rng, out):
    sys.path.insert(0, os.path.join(REPO, "bench", "ifeval_pt"))
    import checkers  # noqa: E402
    prompts = _carregar_jsonl(args.ifeval_prompts)
    kept = 0
    for p in prompts:
        instr = p["turns"][0]; insts = p["instructions"]
        respostas = amostrar(tok, model, device, instr, args.k, args.temperature,
                             args.top_p, args.max_new_tokens)
        passa, falha = [], []
        for r in respostas:
            ok = all(checkers.checar_instrucao(r, ins, loose=False) for ins in insts)
            (passa if ok else falha).append(r)
        if passa and falha:
            out.write(json.dumps({"instruction": instr, "input": "",
                                  "chosen": rng.choice(passa), "rejected": rng.choice(falha),
                                  "meta": {"mode": "instruction", "id": p["id"]}},
                                 ensure_ascii=False) + "\n")
            kept += 1
        print(f"  [instr] {p['id']:<8} passa={len(passa)} falha={len(falha)}")
    print(f"[instruction] pares gerados: {kept}")
    return kept


def modo_quality(args, tok, model, device, rng, out):
    sys.path.insert(0, os.path.join(REPO, "bench", "mtbench_pt"))
    import judge  # noqa: E402
    provider, base, modelo_juiz, key = judge.resolver_provedor()
    if not key:
        print("[quality] defina ANTHROPIC_API_KEY (ou JUDGE_API_KEY). Pulando modo quality.")
        return 0
    print(f"[quality] juiz={provider}/{modelo_juiz}")
    perguntas = _carregar_jsonl(args.mtbench_prompts)
    perguntas = [q for q in perguntas if q.get("category") != "seguranca"]
    kept = 0
    for q in perguntas:
        instr = q["turns"][0]
        respostas = amostrar(tok, model, device, instr, 2, args.temperature,
                             args.top_p, args.max_new_tokens)
        if len(respostas) < 2 or respostas[0] == respostas[1]:
            continue
        notas = []
        for r in respostas:
            rec = {"category": q.get("category", ""), "question": instr, "answer": r,
                   "reference": q.get("reference")}
            saida = judge.chamar_juiz(provider, base, modelo_juiz, key,
                                     judge.JUIZ_SISTEMA, judge.montar_prompt_juiz(rec))
            nota, _ = judge.extrair_nota(saida)
            notas.append(nota if nota is not None else 0)
        if notas[0] == notas[1]:
            continue
        hi, lo = (0, 1) if notas[0] > notas[1] else (1, 0)
        out.write(json.dumps({"instruction": instr, "input": "",
                              "chosen": respostas[hi], "rejected": respostas[lo],
                              "meta": {"mode": "quality", "id": q["id"], "notas": notas}},
                             ensure_ascii=False) + "\n")
        kept += 1
        print(f"  [quality] {q['id']:<14} notas={notas}")
    print(f"[quality] pares gerados: {kept}")
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera pares de preferencia on-policy p/ DPO")
    ap.add_argument("--mode", required=True, choices=["safety", "instruction", "quality"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4, help="amostras por prompt (safety/instruction)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--safety_prompts", default=os.path.join(REPO, "dpo/dpo_seeds/safety_prompts.jsonl"))
    ap.add_argument("--ifeval_prompts", default=os.path.join(REPO, "bench/ifeval_pt/prompts.jsonl"))
    ap.add_argument("--mtbench_prompts", default=os.path.join(REPO, "bench/mtbench_pt/questions.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    try:
        from transformers import set_seed
        set_seed(args.seed)
    except Exception:
        pass

    print(f"[dpo-pairs] modo={args.mode} modelo={args.model}")
    tok, model, device = carregar_modelo(args.model, args.tokenizer, args.attn, args.dtype)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        if args.mode == "safety":
            kept = modo_safety(args, tok, model, device, rng, out)
        elif args.mode == "instruction":
            kept = modo_instruction(args, tok, model, device, rng, out)
        else:
            kept = modo_quality(args, tok, model, device, rng, out)
    print(f"[dpo-pairs] {kept} pares -> {args.out}")
    if args.mode == "safety":
        print("[dpo-pairs] AVISO: 'rejected' contem a obediencia nociva do modelo. "
              "NAO versione este arquivo (dpo_onpolicy/ esta no .gitignore).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
