# Safety PT — sonda de segurança HELD-OUT

Mede se o modelo **recusa pedidos nocivos que ele nunca viu no treino**. É o par
da sonda de over-refusal: juntas medem os dois lados do DPO de segurança —
recusar o nocivo (aqui) sem recusar o benigno (`bench/overrefusal_pt`).

## Prompts (`heldout.jsonl`, 16 itens)

16 pedidos nocivos, um por categoria, com **fraseados e alvos diferentes** dos 50
seeds de treino (`dpo/dpo_seeds/safety_prompts.jsonl`) e dos 6 prompts de
segurança do MT-Bench-PT. Como são disjuntos do treino, a taxa de recusa aqui
mede **generalização**, não decoreba. Sem conteúdo envolvendo menores.

## Métrica

**Taxa de recusa** (quanto MAIOR, melhor — o oposto do over-refusal). Usa a mesma
`eh_recusa` do `dpo/gen_dpo_pairs.py`. Reaproveita o runner da sonda de
over-refusal (`bench/overrefusal_pt/run.py`), só trocando o arquivo de prompts.

## Como rodar (na HPC, GPU, dentro do docker)

```bash
cd /prj/prjgvdc/brunolsm/manaca-1b

# ANTES: instruct-v2 SFT (deve recusar POUCO — o gap que medimos)
MODEL=/data/brunolsm/manaca-checkpoints/manaca-1b-instruct-v2-full LABEL=sft \
  ./scripts/eval/run_overrefusal_pt.sh \
  --prompts bench/safety_pt/heldout.jsonl --out_dir bench/safety_pt/answers

# DEPOIS: o DPO (deve recusar MUITO mais)
MODEL=/data/brunolsm/manaca-checkpoints/manaca-1b-instruct-v2-dpo2-merged LABEL=dpo2 \
  ./scripts/eval/run_overrefusal_pt.sh \
  --prompts bench/safety_pt/heldout.jsonl --out_dir bench/safety_pt/answers
```

## Como ler

- SFT recusa pouco (ex.: 1/16) + DPO recusa muito (ex.: 15/16) = segurança
  melhorou e **generalizou** para pedidos fora do treino. É o resultado alvo.
- As respostas cruas (`answers/`) contêm a obediência nociva do SFT, então **não
  são versionadas** (`.gitignore` cobre `bench/safety_pt/answers/`). Só as taxas
  entram no registro científico.
