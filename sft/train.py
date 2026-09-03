#!/usr/bin/env python3
"""
Manacá-1B — Fase 3a · SFT (Supervised Fine-Tuning) | fiel ao método LLM-jp
=========================================================================
Reproduz o pipeline do `llm-jp-sft` (github.com/llm-jp/llm-jp-sft, plano §8) com
a API MODERNA do trl (SFTConfig + SFTTrainer + DataCollatorForCompletionOnlyLM).

Método (idêntico ao LLM-jp, adaptado para PT-BR conforme a análise
references/phase3-sft):
  - Template Alpaca com preâmbulo + marcadores `### Instrução:` / `### Entrada:`
    / `### Resposta:`, separados por `\\n\\n` (igual ao `### 指示:` / `### 応答:`).
  - Loss SÓ na resposta: o collator mascara todo o prompt com -100
    (DataCollatorForCompletionOnlyLM), computando a perda apenas nos tokens
    depois de `### Resposta:\\n`.
  - Marcadores passados como TOKEN-IDS com o corte `[1:]` que remove o token de
    fronteira do SentencePiece (`▁`): sem isso o collator nunca casa, mascara
    tudo e a loss vira NaN (analise.md §§371-385). Há uma CHECAGEM OBRIGATÓRIA
    do collator antes de treinar.
  - bf16 sempre; gradient checkpointing; cosine + warmup 0.1; max_seq 2048.

Hiperparâmetros FIÉIS ao LLM-jp (README canônico, analise.md §§1119-1146):
  - Full fine-tuning: learning_rate 1e-5, 2 épocas.
  - LoRA:             learning_rate 1e-4, 5 épocas.
  (Definidos automaticamente pelo modo; sobreponíveis via CLI.)

Uso (via wrapper, na imagem manaca-posttrain):
    make sft ARGS="--model_name_or_path menezesbruno/manaca-1b-base \\
        --data_files data/sft/manaca_sft.jsonl \\
        --output_dir /workspace/checkpoints/manaca-1b-instruct2"

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import sys
import types


def _shim_triton_ops() -> None:
    """Compat da imagem: em modo LoRA, o peft importa o bitsandbytes, que (na
    versão da imagem) faz `from triton.ops.matmul_perf_model import ...`. O triton
    3.x removeu `triton.ops`, quebrando o import. Como NÃO usamos quantização
    8-bit, instalamos um stub inócuo para o import passar (essas funções nunca são
    chamadas no caminho bf16/LoRA padrão). No-op no full fine-tuning (sem peft)."""
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

# ── Template Alpaca-PT (igual ao llm-jp-sft, marcadores traduzidos) ───────────
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

# Módulos-alvo LoRA para arquitetura Llama (NÃO usar o preset GPT-NeoX do LLM-jp;
# no Llama, fan_in_fan_out DEVE ser False — analise.md §§972, 1708-1710).
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_formatting(instr_field, input_field, resp_field, eos_token):
    """Formata no template Alpaca-PT e anexa o EOS ao final da resposta (para o
    modelo aprender a PARAR). BATCHED: o trl SFTTrainer chama a formatting_func
    com um batch (dict de listas) e espera uma LISTA de strings de volta."""
    def one(instr, inp, resp):
        instr = (instr or "").strip()
        inp = (inp or "").strip()
        resp = (resp or "").strip()
        if inp:
            prompt = f"{PREAMBLE_INPUT}{INSTRUCTION_TEMPLATE}{instr}{INPUT_TEMPLATE}{inp}"
        else:
            prompt = f"{PREAMBLE_NOINPUT}{INSTRUCTION_TEMPLATE}{instr}"
        return f"{prompt}{RESPONSE_TEMPLATE}{resp}{eos_token}"

    def fmt(examples):
        instrs = examples[instr_field]
        resps = examples[resp_field]
        inputs = examples[input_field] if (input_field and input_field in examples) else None
        return [one(instrs[i], (inputs[i] if inputs is not None else ""), resps[i])
                for i in range(len(instrs))]
    return fmt


def marker_ids(tokenizer, text, slice_boundary=True):
    """Token-ids de um marcador para o collator. Remove o token de fronteira
    inicial do SentencePiece (`▁`) — como o llm-jp-sft (`[1:]`) — para o marcador
    casar no meio da sequência."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if slice_boundary and len(ids) > 1:
        return ids[1:]
    return ids


def check_collator(collator, tokenizer, fmt, sample):
    """Checagem OBRIGATÓRIA (analise.md §§1881-1896): confirma que o collator
    encontra a resposta, mascara o prompt (-100) e deixa a resposta com rótulos
    reais. Levanta erro claro se a máscara sair errada (evita a loss NaN)."""
    text = fmt({k: [v] for k, v in sample.items()})[0]  # fmt é batched -> batch de 1
    enc = tokenizer(text, add_special_tokens=True)
    batch = collator([{"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}])
    labels = batch["labels"][0].tolist()
    n_real = sum(1 for x in labels if x != -100)
    n_mask = sum(1 for x in labels if x == -100)
    if n_real == 0:
        raise RuntimeError(
            "Collator mascarou TODOS os tokens (loss viraria NaN). O marcador "
            f"'{RESPONSE_TEMPLATE!r}' não casou. Verifique a tokenização do marcador "
            "no tokenizador do Manacá (corte [1:] do SentencePiece).")
    if n_mask == 0:
        raise RuntimeError("Collator não mascarou o prompt (loss incidiria no prompt inteiro).")
    print(f"[sft] checagem do collator OK: {n_mask} tokens mascarados (prompt), "
          f"{n_real} com loss (resposta).")


def main() -> int:
    ap = argparse.ArgumentParser(description="Manacá-1B SFT (fiel ao LLM-jp, trl moderno, Alpaca-PT)")
    ap.add_argument("--model_name_or_path", default="menezesbruno/manaca-1b-base")
    ap.add_argument("--tokenizer_name_or_path", default=None)
    ap.add_argument("--dataset_names", default=None, help="IDs HuggingFace, separados por vírgula")
    ap.add_argument("--data_files", nargs="*", default=None, help="Arquivos .jsonl locais (Alpaca)")
    ap.add_argument("--eval_data_files", nargs="*", default=None, help="Arquivos .jsonl de validação (opcional)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--instruction_field", default="instruction")
    ap.add_argument("--input_field", default="input")
    ap.add_argument("--response_field", default="output")
    # Hiperparâmetros. LR/épocas padrão dependem do modo (full vs LoRA), fiéis ao
    # README do llm-jp-sft; use as flags para sobrepor.
    ap.add_argument("--num_train_epochs", type=float, default=None)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--full_finetuning", action="store_true", help="Full FT (LLM-jp flagship; requer ZeRO-3)")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--no_marker_slice", action="store_true",
                    help="Não fatiar [1:] os marcadores (use se a checagem do collator falhar)")
    ap.add_argument("--no_eos", action="store_true", help="Não anexar EOS ao final da resposta")
    ap.add_argument("--no_merge", action="store_true",
                    help="LoRA: NÃO mesclar no base ao final (mantém só o adapter)")
    ap.add_argument("--merged_output_dir", default=None,
                    help="LoRA: destino do modelo COMPLETO mesclado (padrão: <output_dir>-merged)")
    ap.add_argument("--attn", default="flash_attention_2", help="flash_attention_2 | sdpa | eager")
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--resume_from_checkpoint", default=None,
                    help="Retomar treino: 'auto' (último checkpoint-XXX no output_dir) ou um caminho")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_to", default="none")
    args = ap.parse_args()

    # Padrões fiéis ao LLM-jp por modo (README canônico).
    if args.learning_rate is None:
        args.learning_rate = 1e-5 if args.full_finetuning else 1e-4
    if args.num_train_epochs is None:
        args.num_train_epochs = 2.0 if args.full_finetuning else 5.0

    try:
        import torch  # noqa: F401
        from datasets import concatenate_datasets, load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
    except ImportError as e:
        print(f"[ERRO] Dependência ausente ({e}). Rode na imagem 'manaca-posttrain'.")
        return 1

    if not args.dataset_names and not args.data_files:
        print("[ERRO] Forneça --dataset_names e/ou --data_files.")
        return 1

    tok_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = "" if args.no_eos else (tokenizer.eos_token or "")

    def load_many(names, files):
        parts = []
        if names:
            for name in names.split(","):
                parts.append(load_dataset(name.strip(), split="train"))
        if files:
            parts.append(load_dataset("json", data_files=list(files), split="train"))
        return concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    train_ds = load_many(args.dataset_names, args.data_files)
    print(f"[sft] exemplos de treino | training examples: {len(train_ds):,}")
    eval_ds = load_many(None, args.eval_data_files) if args.eval_data_files else None

    fmt = build_formatting(args.instruction_field, args.input_field, args.response_field, eos)

    # ── Modelo | Model ───────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
    )
    model.resize_token_embeddings(len(tokenizer))

    peft_config = None
    if not args.full_finetuning:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGETS, fan_in_fan_out=False, bias="none", task_type="CAUSAL_LM",
        )
        print(f"[sft] LoRA r={args.lora_r} alpha={args.lora_alpha} "
              f"(LR {args.learning_rate}, {args.num_train_epochs} épocas) targets={LORA_TARGETS}")
    else:
        print(f"[sft] Full fine-tuning (LR {args.learning_rate}, {args.num_train_epochs} épocas) — "
              "use accelerate + ZeRO-3 (configs/accelerate_config_zero3.yaml)")

    # Loss só na resposta — collator com marcadores como token-ids (corte [1:] do
    # SentencePiece) + instruction_template para re-mascarar em multi-turno.
    slice_b = not args.no_marker_slice
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=marker_ids(tokenizer, INSTRUCTION_TEMPLATE, slice_b),
        response_template=marker_ids(tokenizer, RESPONSE_TEMPLATE, slice_b),
        tokenizer=tokenizer,
    )
    check_collator(collator, tokenizer, fmt, train_ds[0])  # obrigatória

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        gradient_checkpointing=True,
        # Checkpoint NAO-reentrante: o reentrante (default) reusa parametros em
        # multiplos backwards e quebra o DDP ("marked ready twice"). Compativel
        # com DDP e ZeRO-3. Necessario para o LoRA-SFT rodar em DDP (sft-safety).
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=args.max_seq_length,
        packing=False,          # incompatível com completion-only (mascaramento)
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        report_to=args.report_to,
        eval_strategy=("steps" if eval_ds is not None else "no"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        formatting_func=fmt,
        data_collator=collator,
        peft_config=peft_config,
    )
    # Retomada: 'auto' -> True (o Trainer acha o último checkpoint-XXX no output_dir);
    # um caminho -> retoma daquele checkpoint; None -> começa do zero.
    resume = args.resume_from_checkpoint
    if resume in ("auto", "true", "True", "1", "latest"):
        resume = True
    if resume:
        print(f"[sft] retomando de checkpoint | resuming from: {resume}")
    trainer.train(resume_from_checkpoint=resume)
    # Full FT: save_model grava o modelo COMPLETO. LoRA: grava só o adapter.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[sft] concluído | done -> {args.output_dir}")

    # LoRA -> produz também o modelo COMPLETO (adapter mesclado no base), que é o
    # artefato autossuficiente para avaliar/publicar. Só no processo principal.
    if peft_config is not None and not args.no_merge and trainer.is_world_process_zero():
        merged_dir = args.merged_output_dir or (args.output_dir.rstrip("/") + "-merged")
        try:
            print(f"[sft] mesclando LoRA no base -> modelo completo em {merged_dir}")
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)
            print(f"[sft] modelo COMPLETO salvo | full model saved -> {merged_dir}")
        except Exception as e:  # ex.: pesos shardados sob ZeRO-3 -> use o script standalone
            print(f"[sft] merge em processo falhou ({type(e).__name__}: {e}).")
            print("[sft] rode o merge standalone (1 GPU/CPU):")
            print(f"[sft]   python sft/merge_lora.py --base {args.model_name_or_path} "
                  f"--adapter {args.output_dir} --out {merged_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
