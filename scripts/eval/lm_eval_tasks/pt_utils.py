# -*- coding: utf-8 -*-
"""
Helpers de process_docs para as tarefas PT-BR (exames de multipla escolha).
Usado pelos YAMLs enem_pt / bluex_pt / oab_pt via `!function pt_utils.prep_exame`.

Formato das fontes (eduagarcia): cada exemplo tem
  question   : str (enunciado)
  choices    : {"text": [alternativas...], "label": ["A","B",...]}
  answerKey  : str (letra correta)
  nullified  : bool|None (questoes anuladas devem sair)
Normaliza para:
  query          : enunciado + alternativas rotuladas + "Resposta:"
  choices_label  : lista de letras
  gold           : indice (int) da alternativa correta
"""
LETRAS = ["A", "B", "C", "D", "E", "F", "G"]


def _fmt_exame(doc):
    texts = list(doc["choices"]["text"])
    labels = doc["choices"].get("label") or LETRAS[: len(texts)]
    labels = [str(x) for x in labels]
    linhas = "\n".join(f"{labels[i]}. {texts[i]}" for i in range(len(texts)))
    doc["query"] = f"{str(doc['question']).strip()}\n{linhas}\nResposta:"
    doc["choices_label"] = labels
    try:
        doc["gold"] = labels.index(str(doc["answerKey"]).strip())
    except (ValueError, KeyError):
        doc["gold"] = -1
    return doc


def prep_exame(dataset):
    dataset = dataset.filter(lambda x: not x.get("nullified"))
    dataset = dataset.map(_fmt_exame)
    dataset = dataset.filter(lambda x: x["gold"] >= 0 and len(x["choices_label"]) >= 2)
    return dataset
