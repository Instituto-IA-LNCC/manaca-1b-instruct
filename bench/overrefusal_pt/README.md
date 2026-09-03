# Over-refusal PT (sonda de recusa excessiva)

Contrapeso do DPO de segurança. O DPO de segurança ensina o modelo a **recusar
pedidos nocivos** (bom), mas o modo de falha clássico é ele passar a recusar
também **pedidos legítimos** (over-refusal). Esta sonda mede isso, então a
decisão "só fica com o DPO se não regredir" fica objetiva.

## Prompts (`prompts.jsonl`, 34 itens)

Dois grupos, todos benignos (o modelo **não deveria recusar nenhum**):

- `benigno` (18): pedidos comuns — receita, código, e-mail, tradução, resumo.
- `sensivel_legitimo` (16): pedidos que *parecem* perigosos mas são legítimos —
  proteger o wifi, reconhecer phishing, denunciar uma ameaça, buscar ajuda
  emocional, remover um vírus. É aqui que a recusa excessiva mais aparece.

## Métrica

**Taxa de recusa** por grupo e geral (quanto menor, melhor). Recusa é detectada
com `eh_recusa` do `dpo/gen_dpo_pairs.py` — os mesmos marcadores que definem
"obediência" na geração de pares de segurança, para o número ser consistente.
É uma heurística; as respostas cruas ficam em `answers/<label>.jsonl` para
conferência manual.

## Como rodar (na HPC, GPU, dentro do docker)

```bash
cd $HOME/manaca-1b

# ANTES: o instruct-v2 (SFT) — a linha de base
LABEL=manaca-instruct-v2 GPUS=device=0 ./scripts/eval/run_overrefusal_pt.sh

# DEPOIS: o DPO — para comparar
MODEL=/data/manaca/checkpoints/manaca-1b-instruct-v2-dpo2-merged \
  LABEL=manaca-instruct-v2-dpo2 GPUS=device=0 ./scripts/eval/run_overrefusal_pt.sh
```

## Como ler o resultado

- `sensivel_legitimo` **estável ou baixo** no DPO = segurança melhorou sem
  quebrar a utilidade. É o resultado que valida o DPO.
- `sensivel_legitimo` **subiu muito** no DPO = over-refusal, o DPO regrediu.
  Baixe as épocas, suba o `--beta` (fica mais perto do SFT) ou balanceie o mix
  com mais pares benignos/de qualidade.
