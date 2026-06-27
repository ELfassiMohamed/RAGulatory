"""
rag_conformite.py — RAG Batiment pour Verification Reglementaire
Retrieval TF-IDF natif + LLM. Base : Taxonomie_V-1.xlsx (81 entrees).
"""

import math
import os
import re
import httpx
import pandas as pd
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME, OPENROUTER_BASE_URL

KNOWLEDGE_BASE = []

# ============================================================
# Chargement Taxonomie Batiment depuis Excel
# ============================================================

def _load_taxonomy_base() -> list:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Taxonomie_V-1.xlsx")
    if not os.path.exists(path):
        print("[WARN] Taxonomie_V-1.xlsx introuvable.")
        return []

    xls = pd.ExcelFile(path)
    entries = []

    df = pd.read_excel(xls, "Taxonomie")
    current_domaine = None
    for i, row in df.iterrows():
        if pd.notna(row.get("Domaine")):
            current_domaine = str(row["Domaine"]).strip()
        parts = []
        for col in ["Sous-domaine", "Catégorie", "Caractéristiques"]:
            if pd.notna(row.get(col)):
                parts.append(str(row[col]).strip())
        if not parts:
            continue
        entries.append({
            "id": f"taxo_{i}",
            "source": f"Taxonomie Batiment V1 — {current_domaine or 'Non classe'}",
            "zone": "Reglementation Batiment",
            "text": " — ".join(parts)
        })

    dfg = pd.read_excel(xls, "Glossaire")
    for i, row in dfg.iterrows():
        if pd.isna(row.get("Definition")) or not str(row["Definition"]).strip():
            continue
        text = str(row["Acronyme / Terme"]).strip() if pd.notna(row.get("Acronyme / Terme")) else ""
        if pd.notna(row.get("Domaine")):
            text += f" ({str(row['Domaine']).strip()})"
        text += f" : {str(row['Definition']).strip()}"
        entries.append({
            "id": f"glossaire_{i}",
            "source": "Taxonomie Batiment V1 — Glossaire",
            "zone": "Reglementation Batiment",
            "text": text
        })

    print(f"[OK] Taxonomie chargee : {len(entries)} entrees")
    return entries

KNOWLEDGE_BASE.extend(_load_taxonomy_base())

# ============================================================
# TF-IDF from scratch
# ============================================================

def _tokenize(text: str) -> list:
    return re.findall(r'[a-zA-Záàâäéèêëîïôùûüçñ]{3,}', text.lower())


def _build_tfidf_index(docs: list) -> tuple:
    N = len(docs)
    df = {}
    tok_docs = []
    for doc in docs:
        tokens = _tokenize(doc["text"])
        tok_docs.append(tokens)
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log(N / df[t]) for t in df}
    index = []
    for tokens in tok_docs:
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n = len(tokens) or 1
        index.append({t: (cnt / n) * idf.get(t, 0) for t, cnt in tf.items()})
    return index, idf


def _cosine_similarity(va: dict, vb: dict) -> float:
    common = set(va) & set(vb)
    if not common:
        return 0.0
    dot = sum(va[t] * vb[t] for t in common)
    na = math.sqrt(sum(v ** 2 for v in va.values()))
    nb = math.sqrt(sum(v ** 2 for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


def _vectorize_query(query: str, idf: dict) -> dict:
    tokens = _tokenize(query)
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    n = len(tokens) or 1
    return {t: (cnt / n) * idf.get(t, 0) for t, cnt in tf.items() if t in idf}


# ============================================================
# Moteur RAG (retrieval uniquement)
# ============================================================

class ConformiteRAG:
    def __init__(self, docs=None):
        self.docs = docs or KNOWLEDGE_BASE
        self.tfidf_index, self.idf = _build_tfidf_index(self.docs)
        http_client = httpx.Client(trust_env=False)
        self.client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, http_client=http_client)

    def retrieve(self, query: str, top_k: int = 3, zone_filter: str = None) -> list:
        qv = _vectorize_query(query, self.idf)
        scored = []
        for i, doc in enumerate(self.docs):
            if zone_filter and zone_filter not in (doc.get("zone", ""), "International"):
                continue
            sim = _cosine_similarity(qv, self.tfidf_index[i])
            scored.append({"doc": doc, "score": round(sim, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ============================================================
# Singleton
# ============================================================

_rag_instance = None

def get_rag() -> ConformiteRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = ConformiteRAG()
    return _rag_instance
