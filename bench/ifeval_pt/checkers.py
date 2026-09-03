#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFEval-PT — Checadores de instruções verificáveis (sem juiz)
============================================================
Cada instrução é uma restrição CHECÁVEL POR REGRA na resposta do modelo (no
espírito do IFEval, Zhou et al. 2023, adaptado ao PT-BR). Um checador recebe a
resposta (str) e os kwargs da instrução e devolve True/False.

Avaliação (como no IFEval):
  * strict : verifica na resposta original.
  * loose  : verifica em variantes normalizadas (sem 1a/ultima linha, sem
             markdown, sem cercas de codigo, sem aspas) e passa se QUALQUER
             variante satisfizer. Evita punir por um "Claro! Aqui esta:" na frente.

Só biblioteca padrão.

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import json
import re


# ── utilitarios de contagem ───────────────────────────────────────────────────
def _palavras(t):
    return re.findall(r"\w+", t, flags=re.UNICODE)


def _frases(t):
    return [s for s in re.split(r"[.!?]+", t) if s.strip()]


def _paragrafos(t):
    return [p for p in re.split(r"\n\s*\n", t.strip()) if p.strip()]


def _marcadores(t):
    return [l for l in t.splitlines() if re.match(r"\s*[-*•]\s+\S", l)]


def _itens_numerados(t):
    return [l for l in t.splitlines() if re.match(r"\s*\d+[.)]\s+\S", l)]


def _tem_letra(t):
    return any(c.isalpha() for c in t)


# ── checadores (type -> funcao(resp, kwargs) -> bool) ─────────────────────────
def c_min_palavras(r, k):   return len(_palavras(r)) >= int(k["n"])
def c_max_palavras(r, k):   return len(_palavras(r)) <= int(k["n"])
def c_num_frases(r, k):     return len(_frases(r)) == int(k["n"])
def c_max_frases(r, k):     return len(_frases(r)) <= int(k["n"])
def c_num_paragrafos(r, k): return len(_paragrafos(r)) == int(k["n"])
def c_num_marcadores(r, k): return len(_marcadores(r)) == int(k["n"])
def c_min_marcadores(r, k): return len(_marcadores(r)) >= int(k["n"])
def c_num_itens_num(r, k):  return len(_itens_numerados(r)) == int(k["n"])


def c_incluir_palavra(r, k):
    return re.search(r"\b" + re.escape(k["palavra"]) + r"\b", r, re.IGNORECASE) is not None


def c_incluir_todas(r, k):
    return all(re.search(r"\b" + re.escape(p) + r"\b", r, re.IGNORECASE) for p in k["palavras"])


def c_evitar_palavra(r, k):
    return re.search(r"\b" + re.escape(k["palavra"]) + r"\b", r, re.IGNORECASE) is None


def c_palavra_repetida(r, k):
    n = len(re.findall(r"\b" + re.escape(k["palavra"]) + r"\b", r, re.IGNORECASE))
    return n >= int(k["n"])


def c_tudo_minusculo(r, k):
    return _tem_letra(r) and not any(ch.isupper() for ch in r)


def c_tudo_maiusculo(r, k):
    return _tem_letra(r) and not any(ch.islower() for ch in r)


def c_min_maiusculas(r, k):
    caps = [w for w in _palavras(r) if len(w) >= 2 and w.isupper()]
    return len(caps) >= int(k["n"])


def c_terminar_com(r, k):
    return r.strip().endswith(k["frase"].strip())


def c_comecar_com(r, k):
    return r.strip().startswith(k["frase"].strip())


def c_sem_virgula(r, k):
    return "," not in r


def c_titulo_colchetes(r, k):
    return re.search(r"<<[^>\n]+>>", r) is not None


def c_pos_escrito(r, k):
    return re.search(r"\bp\.?\s?s\.?\b", r, re.IGNORECASE) is not None


def c_min_placeholders(r, k):
    return len(re.findall(r"\[[^\]\n]+\]", r)) >= int(k["n"])


def c_json_valido(r, k):
    s = r.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    m = re.search(r"[{\[].*[}\]]", s, re.DOTALL)
    if m:
        s = m.group(0)
    try:
        json.loads(s)
        return True
    except Exception:
        return False


CHECADORES = {
    "min_palavras": c_min_palavras, "max_palavras": c_max_palavras,
    "num_frases": c_num_frases, "max_frases": c_max_frases,
    "num_paragrafos": c_num_paragrafos,
    "num_marcadores": c_num_marcadores, "min_marcadores": c_min_marcadores,
    "num_itens_numerados": c_num_itens_num,
    "incluir_palavra": c_incluir_palavra, "incluir_todas": c_incluir_todas,
    "evitar_palavra": c_evitar_palavra, "palavra_repetida": c_palavra_repetida,
    "tudo_minusculo": c_tudo_minusculo, "tudo_maiusculo": c_tudo_maiusculo,
    "min_maiusculas": c_min_maiusculas,
    "terminar_com": c_terminar_com, "comecar_com": c_comecar_com,
    "sem_virgula": c_sem_virgula, "titulo_colchetes": c_titulo_colchetes,
    "pos_escrito": c_pos_escrito, "min_placeholders": c_min_placeholders,
    "json_valido": c_json_valido,
}


def _variantes(r):
    """Variantes normalizadas para o modo loose (dedup preservando ordem)."""
    vs = [r]
    linhas = r.split("\n")
    if len(linhas) > 1:
        vs.append("\n".join(linhas[1:]))       # sem 1a linha (preambulo)
        vs.append("\n".join(linhas[:-1]))      # sem ultima linha
        vs.append("\n".join(linhas[1:-1]))     # sem 1a e ultima
    sem_md = r.replace("**", "").replace("__", "").replace("*", "")
    vs.append(sem_md)
    sem_fence = re.sub(r"```[a-zA-Z]*", "", r).replace("```", "")
    vs.append(sem_fence)
    vs.append(r.strip().strip('"\'' + "`"))
    vistos, out = set(), []
    for v in vs:
        if v not in vistos:
            vistos.add(v); out.append(v)
    return out


def checar_instrucao(resp: str, inst: dict, loose: bool = False) -> bool:
    """Checa UMA instrução. loose=True passa se qualquer variante satisfizer."""
    fn = CHECADORES.get(inst["type"])
    if fn is None:
        raise KeyError(f"tipo de instrucao desconhecido: {inst['type']}")
    k = inst.get("kwargs", {})
    candidatos = _variantes(resp) if loose else [resp]
    for cand in candidatos:
        try:
            if fn(cand, k):
                return True
        except Exception:
            continue
    return False
