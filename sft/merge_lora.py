#!/usr/bin/env python3
"""
Manacá-1B — SFT · merge LoRA -> modelo completo | merge LoRA -> full model
==========================================================================
PT
--
Mescla um adapter LoRA no modelo base e salva um **modelo completo,
autossuficiente** (todos os ~1,72B de pesos), pronto para avaliar e publicar no
Hugging Face. É o caminho seguro para obter o modelo inteiro de um treino LoRA
(o `sft/train.py` já tenta mesclar automaticamente ao final; use este script
quando o treino rodou sob DeepSpeed ZeRO-3, onde a mesclagem em processo não é
possível, ou para refazer a mesclagem depois).

Reproduz o resize de embeddings feito no SFT (`resize_token_embeddings`), então o
base e o adapter casam. Full fine-tuning NÃO precisa disto: o `train.py` já grava
o modelo completo direto.

EN
--
Merges a LoRA adapter into the base model and saves a **complete, self-contained
model** (all weights), ready to evaluate and publish. Safe path to obtain the
full model from a LoRA run (train.py already auto-merges when possible; use this
after a ZeRO-3 run or to redo the merge).

Uso | Usage:
    python sft/merge_lora.py \\
        --base menezesbruno/manaca-1b-base \\
        --adapter /workspace/checkpoints/manaca-1b-instruct2 \\
        --out /workspace/checkpoints/manaca-1b-instruct2-merged

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import sys
import types


def _shim_triton_ops() -> None:
    """Compat da imagem: o peft importa o bitsandbytes ao manipular LoRA, e o
    bitsandbytes (na versão da imagem) faz `from triton.ops.matmul_perf_model
    import ...`. O triton 3.x removeu `triton.ops`, quebrando o import. Como não
    usamos quantização 8-bit, instalamos um stub inócuo para o import passar."""
    try:
        import triton  # noqa: F401
    except Exception:
        return
    try:
        import triton.ops  # noqa: F401
        return
    except Exception:
        pass
    ops = types.ModuleType("triton.ops")
    ops.__path__ = []  # type: ignore[attr-defined]
    mpm = types.ModuleType("triton.ops.matmul_perf_model")
    mpm.early_config_prune = lambda *a, **k: None  # type: ignore[attr-defined]
    mpm.estimate_matmul_time = lambda *a, **k: 0.0  # type: ignore[attr-defined]
    ops.matmul_perf_model = mpm  # type: ignore[attr-defined]
    sys.modules.setdefault("triton.ops", ops)
    sys.modules.setdefault("triton.ops.matmul_perf_model", mpm)
    try:
        triton.ops = ops  # type: ignore[attr-defined]
    except Exception:
        pass


_shim_triton_ops()


def main() -> int:
    ap = argparse.ArgumentParser(description="Mescla LoRA no base -> modelo completo (Manacá SFT)")
    ap.add_argument("--base", required=True, help="modelo base (ex.: menezesbruno/manaca-1b-base)")
    ap.add_argument("--adapter", required=True, help="diretório do adapter LoRA (output_dir do train.py)")
    ap.add_argument("--out", required=True, help="destino do modelo COMPLETO mesclado")
    ap.add_argument("--tokenizer", default=None, help="tokenizer (padrão: o salvo no --adapter)")
    ap.add_argument("--merge_dtype", default="float32",
                    help="precisão da MESCLAGEM W+ΔW (float32 evita arredondar deltas pequenos do LoRA)")
    ap.add_argument("--save_dtype", default="bfloat16",
                    help="precisão de GRAVAÇÃO do modelo mesclado (bfloat16 | float16 | float32)")
    ap.add_argument("--dtype", default=None, help="(compat) atalho: define merge_dtype e save_dtype juntos")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"[ERRO] Dependência ausente ({e}). Rode na imagem 'manaca-posttrain'.")
        return 1

    if args.dtype:  # compat com chamadas antigas
        args.merge_dtype = args.save_dtype = args.dtype
    merge_dtype = getattr(torch, args.merge_dtype, torch.float32)
    save_dtype = getattr(torch, args.save_dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.adapter)

    # Mescla em fp32 (ou merge_dtype) e SÓ ENTÃO baixa para save_dtype: fazer o
    # W+ΔW direto em bf16 arredonda deltas pequenos do LoRA (bf16 tem ~3 dígitos),
    # podendo apagar o efeito do treino. Ver dpo/diag_dpo.py.
    print(f"[merge] base={args.base} adapter={args.adapter} merge_dtype={args.merge_dtype} save_dtype={args.save_dtype}")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=merge_dtype)
    # Casa o tamanho da embedding com o do SFT (o train.py faz resize_token_embeddings).
    if base.get_input_embeddings().weight.shape[0] != len(tok):
        base.resize_token_embeddings(len(tok))

    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()          # W+ΔW em merge_dtype (preciso)
    if save_dtype != merge_dtype:
        model = model.to(save_dtype)          # baixa a precisão só p/ gravar
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"[merge] modelo COMPLETO salvo | full model saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
