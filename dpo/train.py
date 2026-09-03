#!/usr/bin/env python3
"""
Manacá-1B — Fase 3b · DPO (Direct Preference Optimization) | fiel ao llm-jp-dpo
==============================================================================
Espelha o `llm-jp-dpo` com a API MODERNA do trl (DPOConfig/DPOTrainer). Roda por
CIMA do modelo do SFT (o instruct) e alinha o comportamento com preferências.

Hiperparâmetros FIÉIS ao LLM-jp: beta=0.1, LR=5e-7 (20x menor que o SFT),
cosine + warmup 0.1, bf16, gradient checkpointing, max_length 2048 /
max_prompt_length 1024.

Dados (esquema): `{instruction, input?, chosen, rejected}` — o prompt recebe o
MESMO template Alpaca-PT do SFT; EOS é anexado a chosen/rejected. Gere com
`sft/prepare_dpo.py`.

SALVA O MODELO COMPLETO (como o SFT):
  - full FT: `save_model` grava o modelo inteiro em --output_dir.
  - LoRA (padrão; cabe em 2x24GB pois a referência = base sem adapter): mescla o
    adapter no base e salva o modelo completo em <output_dir>-merged.

Uso (via wrapper, na imagem manaca-posttrain):
    make dpo ARGS="--model_name_or_path /workspace/checkpoints/manaca-1b-instruct-v2-full \\
        --data_files data/dpo/manaca_dpo.jsonl \\
        --output_dir /workspace/checkpoints/manaca-1b-instruct-v2-dpo"

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import os
import sys
import types


def _shim_triton_ops() -> None:
    """Compat da imagem: ao criar o LoRA, o peft importa o bitsandbytes, que (na
    versão da imagem) faz `from triton.ops.matmul_perf_model import ...`. O triton
    3.x removeu `triton.ops`, quebrando o import. Como NÃO usamos quantização
    8-bit, instalamos um stub inócuo para o import passar (essas funções nunca são
    chamadas no caminho bf16/LoRA padrão)."""
    try:
        import triton  # noqa: F401
    except Exception:
        return  # sem triton, nada a fazer
    try:
        import triton.ops  # noqa: F401
        return  # já existe -> imagem ok, não mexe
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
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _s(x):
    return ("" if x is None else str(x)).strip()


def build_map(instr_f, input_f, chosen_f, rejected_f, eos):
    """Constrói {prompt, chosen, rejected} no template Alpaca-PT (prompt idêntico
    ao SFT; EOS anexado às respostas para consolidar o 'saber parar')."""
    def fmt(ex):
        instr = _s(ex.get(instr_f))
        inp = _s(ex.get(input_f)) if input_f else ""
        if inp:
            prompt = f"{PREAMBLE_INPUT}{INSTRUCTION_TEMPLATE}{instr}{INPUT_TEMPLATE}{inp}{RESPONSE_TEMPLATE}"
        else:
            prompt = f"{PREAMBLE_NOINPUT}{INSTRUCTION_TEMPLATE}{instr}{RESPONSE_TEMPLATE}"
        return {"prompt": prompt,
                "chosen": _s(ex.get(chosen_f)) + eos,
                "rejected": _s(ex.get(rejected_f)) + eos}
    return fmt


def main() -> int:
    ap = argparse.ArgumentParser(description="Manacá-1B DPO (fiel ao llm-jp-dpo, trl moderno)")
    ap.add_argument("--model_name_or_path", default="/workspace/checkpoints/manaca-1b-instruct-v2-full",
                    help="modelo do SFT (instruct) sobre o qual rodar o DPO")
    ap.add_argument("--tokenizer_name_or_path", default=None)
    ap.add_argument("--dataset_names", default=None, help="IDs HuggingFace, separados por vírgula")
    ap.add_argument("--data_files", nargs="*", default=None, help="Arquivos .jsonl locais de preferência")
    ap.add_argument("--output_dir", default="/workspace/checkpoints/manaca-1b-instruct-v2-dpo")
    ap.add_argument("--instruction_field", default="instruction")
    ap.add_argument("--input_field", default="input")
    ap.add_argument("--chosen_field", default="chosen")
    ap.add_argument("--rejected_field", default="rejected")
    # Hiperparâmetros fiéis ao LLM-jp
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--learning_rate", type=float, default=5e-7)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--full_finetuning", action="store_true", help="Full FT (2 cópias na memória; requer ZeRO-3 + folga)")
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--no_eos", action="store_true", help="Não anexar EOS a chosen/rejected")
    ap.add_argument("--no_merge", action="store_true", help="LoRA: não mesclar no base ao final")
    ap.add_argument("--merged_output_dir", default=None, help="LoRA: destino do modelo COMPLETO mesclado")
    ap.add_argument("--attn", default="flash_attention_2", help="flash_attention_2 | sdpa | eager")
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--save_strategy", default="steps", choices=["steps", "epoch", "no"],
                    help="Quando salvar: 'steps' (a cada --save_steps) | 'epoch' (fim de cada época) | 'no'")
    ap.add_argument("--save_total_limit", type=int, default=3,
                    help="Máx. de checkpoints mantidos (use alto, ex.: 30, p/ comparar épocas)")
    ap.add_argument("--resume_from_checkpoint", default=None,
                    help="Retomar treino: 'auto' (último checkpoint-XXX no output_dir) ou um caminho")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_to", default="none")
    args = ap.parse_args()

    # ── Guardas amigáveis: o DPO é o ÚLTIMO passo ─────────────────────────────
    # Roda DEPOIS que o SFT v2 termina (modelo em disco) e COM pares reais de
    # preferência. Falha cedo, com mensagem clara, em vez de um traceback do HF.
    def _is_local(p: str) -> bool:
        return p.startswith(("/", "./", "../"))

    if _is_local(args.model_name_or_path) and not os.path.isdir(args.model_name_or_path):
        print(f"[ERRO] modelo do SFT não encontrado: {args.model_name_or_path}\n"
              "       O DPO só roda DEPOIS que o SFT termina. Confira se o treino do\n"
              "       SFT v2 concluiu e o diretório existe (docker ps / tail no log).")
        return 1
    if args.data_files:
        faltando = [f for f in args.data_files if not os.path.isfile(f)]
        if faltando:
            print(f"[ERRO] arquivo(s) de preferência ausente(s): {faltando}\n"
                  "       Gere os pares ANTES: make dpo-data ARGS=\"--dataset <hf_id_real>\"\n"
                  "       (ou --data_files pares.jsonl). Um literal '<hf_id>' NÃO é dataset.")
            return 1

    try:
        import torch
        from datasets import concatenate_datasets, load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        print(f"[ERRO] Dependência ausente ({e}). Rode na imagem 'manaca-posttrain'.")
        return 1

    if not args.dataset_names and not args.data_files:
        print("[ERRO] Forneça --dataset_names e/ou --data_files (pares de preferência). "
              "Gere com sft/prepare_dpo.py.")
        return 1

    tok_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = "" if args.no_eos else (tokenizer.eos_token or "")

    parts = []
    if args.dataset_names:
        for name in args.dataset_names.split(","):
            parts.append(load_dataset(name.strip(), split="train"))
    if args.data_files:
        parts.append(load_dataset("json", data_files=list(args.data_files), split="train"))
    train_ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    fmt = build_map(args.instruction_field, args.input_field, args.chosen_field, args.rejected_field, eos)
    train_ds = train_ds.map(fmt, remove_columns=train_ds.column_names)
    print(f"[dpo] pares de preferência | preference pairs: {len(train_ds):,}")

    # ── Modelo | Model ───────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
    )

    peft_config = None
    ref_model = None
    if not args.full_finetuning:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGETS, fan_in_fan_out=False, bias="none", task_type="CAUSAL_LM",
        )
        print(f"[dpo] LoRA r={args.lora_r} (beta={args.beta}, LR={args.learning_rate}, "
              f"{args.num_train_epochs} épocas) — referência = base sem adapter (leve)")
    else:
        # Full FT: a referência precisa ser uma cópia congelada (pesado em memória).
        print(f"[dpo] Full FT (beta={args.beta}, LR={args.learning_rate}) — referência = cópia congelada; "
              "requer ZeRO-3 e bastante memória")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
        )

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=args.report_to,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,          # None + LoRA -> referência = base sem adapter
        args=dpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    resume = args.resume_from_checkpoint
    if resume in ("auto", "true", "True", "1", "latest"):
        resume = True
    if resume:
        print(f"[dpo] retomando de checkpoint | resuming from: {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[dpo] concluído | done -> {args.output_dir}")

    # LoRA -> modelo COMPLETO mesclado (autossuficiente), como no SFT.
    if peft_config is not None and not args.no_merge and trainer.is_world_process_zero():
        merged_dir = args.merged_output_dir or (args.output_dir.rstrip("/") + "-merged")
        try:
            print(f"[dpo] mesclando LoRA no base -> modelo completo em {merged_dir}")
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)
            print(f"[dpo] modelo COMPLETO salvo | full model saved -> {merged_dir}")
        except Exception as e:
            print(f"[dpo] merge em processo falhou ({type(e).__name__}: {e}).")
            print(f"[dpo] rode: python sft/merge_lora.py --base {args.model_name_or_path} "
                  f"--adapter {args.output_dir} --out {merged_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
