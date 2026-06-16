"""
rag_conformite.py — RAG léger pour l'Agent Conformité
Retrieval-Augmented Generation avec TF-IDF natif Python + LLM.

Pipeline :
  1. Indexation : chunks de documents juridiques → vecteurs TF-IDF
  2. Requête : question vectorisée → similarité cosinus → top-k chunks
  3. Génération : chunks + question → LLM → rapport sourcé

Pas de dépendance externe (ChromaDB, Pinecone, sklearn) :
  implémentation TF-IDF from scratch avec math standard.
"""

import math
import os
import re
import httpx
import pandas as pd
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME, OPENROUTER_BASE_URL

# ============================================================
# Base de connaissances : CNDP (Maroc) + RGPD (UE)
# ============================================================

KNOWLEDGE_BASE = [
    # ── CNDP Maroc ──────────────────────────────────────────
    {
        "id": "cndp_001",
        "source": "CNDP Maroc — Loi 09-08, Art. 3",
        "zone": "Maroc",
        "text": (
            "La Commission Nationale de contrôle de la protection des Données à "
            "caractère Personnel (CNDP) exige que tout traitement de données "
            "personnelles au Maroc fasse l'objet d'une déclaration ou d'une "
            "autorisation préalable. Les entreprises doivent déclarer leurs "
            "traitements automatisés de données personnelles auprès de la CNDP "
            "avant toute mise en œuvre."
        )
    },
    {
        "id": "cndp_002",
        "source": "CNDP Maroc — Loi 09-08, Art. 18 (Transferts internationaux)",
        "zone": "Maroc",
        "text": (
            "Le transfert de données personnelles vers un pays étranger n'est "
            "autorisé que si ce pays assure un niveau de protection adéquat. "
            "Pour un SaaS hébergé hors Maroc (Europe, USA), l'entreprise doit "
            "obtenir une autorisation spéciale de la CNDP ou s'assurer que "
            "l'éditeur dispose de clauses contractuelles types. Hébergement sur "
            "des serveurs au Maroc ou en Union Européenne recommandé."
        )
    },
    {
        "id": "cndp_003",
        "source": "CNDP Maroc — Obligations de sécurité",
        "zone": "Maroc",
        "text": (
            "Le responsable du traitement doit prendre toutes les précautions "
            "utiles pour préserver la sécurité des données. Cela inclut : "
            "le chiffrement des données au repos et en transit, "
            "l'authentification multi-facteurs (MFA) pour les accès, "
            "la journalisation des accès, et un plan de reprise après sinistre. "
            "Les SaaS doivent fournir un rapport de sécurité annuel."
        )
    },
    # ── RGPD Union Européenne ────────────────────────────────
    {
        "id": "rgpd_001",
        "source": "RGPD (UE) 2016/679 — Art. 5 (Principes)",
        "zone": "UE",
        "text": (
            "Le RGPD impose 6 principes fondamentaux : licéité/loyauté/transparence, "
            "limitation des finalités, minimisation des données, exactitude, "
            "limitation de la conservation, intégrité et confidentialité. "
            "Tout SaaS traitant des données de citoyens européens doit respecter "
            "ces principes, quelle que soit la localisation du serveur."
        )
    },
    {
        "id": "rgpd_002",
        "source": "RGPD (UE) 2016/679 — Art. 28 (Sous-traitants)",
        "zone": "UE",
        "text": (
            "Un SaaS agissant comme sous-traitant doit signer un DPA "
            "(Data Processing Agreement) avec l'entreprise cliente. "
            "Ce contrat doit préciser : la nature du traitement, sa durée, "
            "les instructions de traitement, les mesures de sécurité, "
            "et les conditions de suppression/restitution des données "
            "à la fin du contrat (droit à la portabilité et réversibilité)."
        )
    },
    {
        "id": "rgpd_003",
        "source": "RGPD (UE) 2016/679 — Art. 46 (Transferts vers pays tiers)",
        "zone": "UE",
        "text": (
            "Les transferts de données personnelles vers des pays hors UE "
            "(ex: USA, Maroc sans décision d'adéquation) nécessitent des "
            "garanties appropriées : clauses contractuelles types (CCT) "
            "approuvées par la Commission, règles d'entreprise contraignantes (BCR), "
            "ou certification ISO 27001 du sous-traitant. Le Privacy Shield "
            "USA-UE a été invalidé en 2020 (arrêt Schrems II)."
        )
    },
    # ── SLA et sécurité technique ────────────────────────────
    {
        "id": "sla_001",
        "source": "Bonnes pratiques SLA — Disponibilité",
        "zone": "International",
        "text": (
            "SLA 99.0% = 87.6h de downtime/an (inacceptable pour systèmes critiques). "
            "SLA 99.9% = 8.7h de downtime/an (standard professionnel minimum). "
            "SLA 99.95% = 4.4h/an (recommandé pour PME critiques). "
            "SLA 99.99% = 52min/an (entreprises/données sensibles). "
            "Exiger une pénalité financière (SLA crédit) en cas de non-respect : "
            "10-25% du mensuel pour chaque % de disponibilité manquant."
        )
    },
    {
        "id": "sla_002",
        "source": "Bonnes pratiques — Réversibilité et portabilité des données",
        "zone": "International",
        "text": (
            "La clause de réversibilité est un point critique souvent omis. "
            "Elle doit préciser : le délai de restitution des données (max 30 jours), "
            "le format d'export (CSV, JSON, SQL), la période de rétention post-résiliation "
            "(minimum 90 jours), et les frais éventuels d'export. "
            "Sans cette clause, l'entreprise risque le vendor lock-in total. "
            "Les SaaS certifiés ISO 27001 incluent généralement cette garantie."
        )
    },
    {
        "id": "security_001",
        "source": "Certifications de sécurité SaaS",
        "zone": "International",
        "text": (
            "ISO 27001 : Norme internationale de management de la sécurité de l'information. "
            "SOC 2 Type II : Audit indépendant américain sur la sécurité, disponibilité et confidentialité. "
            "HDS (Hébergement Données de Santé) : Obligatoire pour les données médicales en France. "
            "PCI-DSS : Requis pour les données de cartes bancaires. "
            "Un SaaS B2B sérieux doit présenter au minimum ISO 27001 ou SOC 2 Type II. "
            "Demander systématiquement le rapport d'audit le plus récent à l'éditeur."
        )
    }
]

# ============================================================
# Chargement Taxonomie Bâtiment depuis Excel
# ============================================================

def _load_taxonomy_base() -> list:
    """Charge Taxonomie_V-1.xlsx et convertit en format KNOWLEDGE_BASE."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Taxonomie_V-1.xlsx")
    if not os.path.exists(path):
        print("[WARN] Taxonomie_V-1.xlsx introuvable — ignoré.")
        return []

    xls = pd.ExcelFile(path)
    entries = []

    # Sheet 1 : Taxonomie
    df_taxo = pd.read_excel(xls, "Taxonomie")
    current_domaine = None
    for i, row in df_taxo.iterrows():
        domaine = row.get("Domaine")
        if pd.notna(domaine):
            current_domaine = str(domaine).strip()

        sous_domaine = row.get("Sous-domaine")
        categorie = row.get("Catégorie")
        caracteristiques = row.get("Caractéristiques")

        parts = []
        if pd.notna(sous_domaine):
            parts.append(str(sous_domaine).strip())
        if pd.notna(categorie):
            parts.append(str(categorie).strip())
        if pd.notna(caracteristiques):
            parts.append(str(caracteristiques).strip())
        if not parts:
            continue

        entries.append({
            "id": f"taxo_{i}",
            "source": f"Taxonomie Bâtiment V1 — {current_domaine or 'Non classé'}",
            "zone": "Réglementation Bâtiment",
            "text": " — ".join(parts)
        })

    # Sheet 2 : Glossaire
    df_gloss = pd.read_excel(xls, "Glossaire")
    for i, row in df_gloss.iterrows():
        acronyme = row.get("Acronyme / Terme")
        domaine = row.get("Domaine")
        definition = row.get("Définition")

        if pd.isna(definition) or not str(definition).strip():
            continue

        text = str(acronyme).strip() if pd.notna(acronyme) else ""
        if pd.notna(domaine):
            text += f" ({str(domaine).strip()})"
        text += f" : {str(definition).strip()}"

        entries.append({
            "id": f"glossaire_{i}",
            "source": "Taxonomie Bâtiment V1 — Glossaire",
            "zone": "Réglementation Bâtiment",
            "text": text
        })

    print(f"[OK] Taxonomie chargée : {len(entries)} entrées")
    return entries

# Fusion dans la base de connaissances
KNOWLEDGE_BASE.extend(_load_taxonomy_base())

# ============================================================
# TF-IDF from scratch
# ============================================================

def _tokenize(text: str) -> list:
    """Tokenisation simple : minuscules, mots de 3+ caractères."""
    return re.findall(r'[a-záàâäéèêëîïôùûüçñ]{3,}', text.lower())


def _build_tfidf_index(docs: list) -> tuple:
    """
    Construit l'index TF-IDF.
    Retourne (index, idf_dict) où index[i] = {term: tf-idf weight}.
    """
    # IDF : log(N / df_t)
    N = len(docs)
    df = {}
    tokenized_docs = []

    for doc in docs:
        tokens = _tokenize(doc["text"])
        tokenized_docs.append(tokens)
        unique_tokens = set(tokens)
        for t in unique_tokens:
            df[t] = df.get(t, 0) + 1

    idf = {t: math.log(N / df[t]) for t in df}

    # TF-IDF par document
    tfidf_index = []
    for tokens in tokenized_docs:
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n = len(tokens) or 1
        vec = {t: (cnt / n) * idf.get(t, 0) for t, cnt in tf.items()}
        tfidf_index.append(vec)

    return tfidf_index, idf


def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Similarité cosinus entre deux vecteurs TF-IDF."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _vectorize_query(query: str, idf: dict) -> dict:
    """Vectorise une requête avec le même IDF que le corpus."""
    tokens = _tokenize(query)
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    n = len(tokens) or 1
    return {t: (cnt / n) * idf.get(t, 0) for t, cnt in tf.items() if t in idf}


# ============================================================
# Moteur RAG
# ============================================================

class ConformiteRAG:
    """
    RAG léger pour l'Agent Conformité.
    Retrieval par TF-IDF + Génération par LLM.
    """

    def __init__(self, docs=None):
        self.docs = docs or KNOWLEDGE_BASE
        self.tfidf_index, self.idf = _build_tfidf_index(self.docs)
        http_client = httpx.Client(trust_env=False)
        self.client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, http_client=http_client)

    def retrieve(self, query: str, top_k: int = 3, zone_filter: str = None) -> list:
        """
        Récupère les top_k chunks les plus pertinents.

        Args:
            query       : Question de conformité
            top_k       : Nombre de chunks à retourner
            zone_filter : "Maroc" | "UE" | "Réglementation Bâtiment" | None (tous)

        Returns:
            Liste de dicts {"doc", "score"}
        """
        query_vec = _vectorize_query(query, self.idf)
        scored = []

        for i, doc in enumerate(self.docs):
            if zone_filter and zone_filter not in (doc.get("zone", ""), "International"):
                continue
            sim = _cosine_similarity(query_vec, self.tfidf_index[i])
            scored.append({"doc": doc, "score": round(sim, 4)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def generate(self, query: str, zone: str = None, lang: str = "fr") -> dict:
        """
        Pipeline RAG complet : retrieve → augment → generate.

        Returns:
            dict {"answer": str, "sources": list, "chunks_used": int}
        """
        # 1. Retrieve
        chunks = self.retrieve(query, top_k=3, zone_filter=zone)
        if not chunks:
            chunks = self.retrieve(query, top_k=3, zone_filter=None)

        # 2. Construire le contexte
        context_parts = []
        sources = []
        for item in chunks:
            doc = item["doc"]
            context_parts.append(f"[{doc['source']}]\n{doc['text']}")
            sources.append({"source": doc["source"], "zone": doc["zone"], "score": item["score"]})

        context = "\n\n---\n\n".join(context_parts)

        lang_rule = "Réponds UNIQUEMENT en FRANÇAIS." if lang == "fr" else "Respond ONLY in ENGLISH."

        system = f"""Tu es l'Agent Conformité, expert en réglementation (SaaS, RGPD, CNDP, sécurité, bâtiment).
{lang_rule}

Utilise les extraits réglementaires fournis pour répondre à la question.
Structure ta réponse avec :
1. Un verdict de risque : ✅ Faible / ⚠️ Moyen / 🔴 Élevé
2. Les points réglementaires clés
3. Les questions à poser à l'éditeur SaaS / au professionnel
4. Les recommandations concrètes

Ne cite jamais les numéros d'article directement, mais synthétise les obligations en langage accessible.
"""

        user_msg = f"""Contexte réglementaire disponible :
{context}

---

Question de conformité : {query}"""

        # 3. Generate
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.3,
                max_tokens=800
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Erreur de génération : {e}"

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks)
        }


# ============================================================
# Singleton
# ============================================================

_rag_instance = None

def get_rag() -> ConformiteRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = ConformiteRAG()
    return _rag_instance


def rag_query(query: str, zone: str = None, lang: str = "fr") -> dict:
    """Point d'entrée simplifié."""
    return get_rag().generate(query, zone=zone, lang=lang)


# ============================================================
# Classification Taxonomie Bâtiment
# ============================================================

def classify_building(query: str, top_k: int = 5) -> dict:
    """
    Classifie un bâtiment à partir d'une description textuelle.
    Utilise la similarité TF-IDF contre la base Taxonomie Bâtiment.
    
    Returns:
        dict avec { "query", "classifications": [...], "tree_link" }
    """
    rag = get_rag()
    chunks = rag.retrieve(query, top_k=top_k, zone_filter="Réglementation Bâtiment")
    
    results = []
    for c in chunks:
        doc = c["doc"]
        parts = doc["text"].split(" — ")
        results.append({
            "id": doc["id"],
            "source": doc["source"],
            "score": c["score"],
            "sous_domaine": parts[0] if len(parts) > 0 else "",
            "categorie": parts[1] if len(parts) > 1 else "",
            "caracteristiques": parts[2] if len(parts) > 2 else "",
            "full_text": doc["text"]
        })
    
    return {
        "query": query,
        "classifications": results,
        "count": len(results)
    }


def get_taxonomy_tree() -> dict:
    """
    Retourne la taxonomie sous forme d'arbre structuré :
    Domaine → Sous-domaine → [{ Catégorie, Caractéristiques }].
    """
    tree = {}
    for doc in KNOWLEDGE_BASE:
        if not doc["id"].startswith("taxo_"):
            continue
        parts = doc["text"].split(" — ")
        domaine = doc["source"].replace("Taxonomie Bâtiment V1 — ", "")
        
        if domaine not in tree:
            tree[domaine] = {}
        
        sous_domaine = parts[0] if len(parts) > 0 else ""
        categorie = parts[1] if len(parts) > 1 else ""
        caracteristiques = parts[2] if len(parts) > 2 else ""
        
        if sous_domaine not in tree[domaine]:
            tree[domaine][sous_domaine] = []
        
        tree[domaine][sous_domaine].append({
            "categorie": categorie,
            "caracteristiques": caracteristiques,
            "id": doc["id"]
        })
    
    return {
        "domaines": [
            {
                "nom": domaine,
                "sous_domaines": [
                    {
                        "nom": sd,
                        "entrees": entries
                    }
                    for sd, entries in sous_domaines.items()
                ]
            }
            for domaine, sous_domaines in tree.items()
        ],
        "count_entries": sum(
            len(entries)
            for sd_dict in tree.values()
            for entries in sd_dict.values()
        )
    }


# ============================================================
# Test rapide
# ============================================================

if __name__ == "__main__":
    rag = ConformiteRAG()

    queries = [
        ("Mon SaaS est hébergé aux USA, est-ce conforme pour le Maroc ?", "Maroc"),
        ("Quelles certifications de sécurité dois-je exiger à l'éditeur ?", None),
        ("Que doit contenir la clause de réversibilité dans mon contrat SaaS ?", None)
    ]

    for q, zone in queries:
        print(f"\nQuestion : {q}")
        chunks = rag.retrieve(q, top_k=2, zone_filter=zone)
        print(f"  Top chunks récupérés :")
        for c in chunks:
            print(f"    [{c['score']:.3f}] {c['doc']['source']}")
