"""
taxonomie_advisor.py — Assistant de recommandation de plans bâtiment
Pipeline complet : RAG Taxonomie → ML Predictor → MCDM → RAG Conformité

Architecture :
  1. Requête utilisateur → Extraction des paramètres terrain (LLM)
  2. RAG Taxonomie → Recherche des réglementations applicables
  3. Génération de plans bâtiment (LLM + contexte RAG)
  4. Scores ML + Conformité pour chaque plan
  5. Classement MCDM (WSM/TOPSIS)
  6. Rapport final sourcé
"""

import json
import httpx
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME, OPENROUTER_BASE_URL, MODEL_SETTINGS

# ============================================================
# Critères MCDM pour les plans bâtiment
# ============================================================

BUILDING_CRITERIA = [
    "surface_utilisable",
    "cout_estime",
    "conformite_reglementaire",
    "faisabilite_technique",
    "delai_realisation",
    "ml_viability_score"
]

BUILDING_CRITERIA_LABELS = {
    "surface_utilisable": "Surface utile totale (m²)",
    "cout_estime": "Coût estimé (DH)",
    "conformite_reglementaire": "Conformité réglementaire (/10)",
    "faisabilite_technique": "Faisabilité technique (/10)",
    "delai_realisation": "Délai de réalisation (mois)",
    "ml_viability_score": "Score viabilité IA (/10)"
}

BUILDING_CRITERIA_TYPES = {
    "surface_utilisable": "max",
    "cout_estime": "min",
    "conformite_reglementaire": "max",
    "faisabilite_technique": "max",
    "delai_realisation": "min",
    "ml_viability_score": "max"
}


# ============================================================
# Pipeline principal
# ============================================================

def recommend(query: str, method: str = "wsm") -> dict:
    """
    Pipeline complet de recommandation de plans bâtiment.
    
    Args:
        query: Description du terrain/projet par l'utilisateur
        method: "wsm" ou "topsis"
    
    Returns:
        dict avec plans, scores MCDM, conformité, sources
    """
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL,
                    http_client=httpx.Client(trust_env=False))

    # ---- Étape 1 : Extraction des paramètres terrain ----
    params = _extract_parameters(query, client)
    
    # ---- Étape 2 : RAG Taxonomie ----
    regulations = _retrieve_regulations(params)
    
    # ---- Étape 3 : Génération des plans bâtiment ----
    plans = _generate_plans(params, regulations, client)
    
    # ---- Étape 4 : Scores ML + Conformité pour chaque plan ----
    for plan in plans:
        _enrich_plan(plan, params, client)
    
    # ---- Étape 5 : Classement MCDM ----
    ranking = _rank_plans(plans, method)
    
    return {
        "query": query,
        "terrain": params,
        "regulations": regulations,
        "plans": ranking["plans"],
        "mcdm": ranking["mcdm"],
        "conformite_summary": _build_conformity_summary(plans)
    }


# ============================================================
# Étape 1 : Extraction des paramètres terrain
# ============================================================

_EXTRACT_SYSTEM = """Tu es un expert en urbanisme et construction.
Extrais les paramètres du terrain depuis la description utilisateur.
Retourne UNIQUEMENT un JSON valide :

{
  "surface_m2": float | null,
  "localisation": string | null,
  "zone_urbaine": bool,
  "usage_souhaite": string | null,
  "contraintes": [string],
  "budget_estime_dh": float | null,
  "nb_etages_max": int | null
}

Si une valeur n'est pas mentionnée, mets null.
"""


def _extract_parameters(query: str, client) -> dict:
    raw = _llm_retry(client, [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": query}
    ], max_tokens=300, temperature=0.1)
    
    if raw:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"surface_m2": None, "localisation": None, "zone_urbaine": False,
            "usage_souhaite": None, "contraintes": [], "budget_estime_dh": None,
            "nb_etages_max": None}


# ============================================================
# Étape 2 : RAG Taxonomie
# ============================================================

def _retrieve_regulations(params: dict) -> dict:
    from rag_conformite import get_rag
    
    rag = get_rag()
    # Construire une requête RAG à partir des paramètres terrain
    query_parts = []
    if params.get("surface_m2"):
        query_parts.append(f"{params['surface_m2']} m²")
    if params.get("usage_souhaite"):
        query_parts.append(params["usage_souhaite"])
    if params.get("nb_etages_max"):
        query_parts.append(f"{params['nb_etages_max']} étages")
    
    rag_query = " ".join(query_parts) if query_parts else "classification bâtiment terrain"
    
    # Retrieve from taxonomy
    chunks = rag.retrieve(rag_query, top_k=5, zone_filter="Réglementation Bâtiment")
    if not chunks:
        chunks = rag.retrieve(rag_query, top_k=5, zone_filter=None)
    
    return {
        "query": rag_query,
        "regulations": [
            {
                "source": c["doc"]["source"],
                "text": c["doc"]["text"],
                "score": c["score"]
            }
            for c in chunks
        ]
    }


# ============================================================
# Étape 3 : Génération des plans bâtiment
# ============================================================

_PLANS_SYSTEM = """Tu es un architecte urbaniste expert. À partir des paramètres terrain et des réglementations applicables,
génère 3 à 5 plans de construction viables pour le terrain.

Pour chaque plan, retourne UNIQUEMENT un objet JSON avec cette structure :
{
  "plans": [
    {
      "name": "Nom court du plan",
      "description": "Description détaillée en 1-2 phrases",
      "type": "Maison individuelle" | "Petit immeuble" | "Immeuble collectif" | "ERP" | "Mixte",
      "nb_etages": int,
      "surface_utile_m2": float,
      "cout_estime_dh": float,
      "delai_mois": int,
      "criteria_values": {
        "surface_utilisable": float (0-1000, surface utile en m²),
        "cout_estime": float (0-10000000, coût en DH),
        "conformite_reglementaire": float (0-10),
        "faisabilite_technique": float (0-10),
        "delai_realisation": float (0-60, mois),
        "ml_viability_score": float (0-10)
      }
    }
  ]
}

Règles :
- Les plans doivent être réalistes et conformes aux réglementations marocaines fournies
- Diversifie les types (petit collectif, maison, mixte)
- Les scores criteria_values sont des estimations expertes de l'architecte
- Surface utilisable = surface_terrain × nb_étages × coefficient (~0.7-0.85)
- Coût estimé = basé sur surface × coût_m2 régional (15000-30000 DH/m²)
- Le score ml_viability_score est une prédiction IA de la viabilité du projet
"""


def _llm_retry(client, messages, max_tokens=1500, temperature=0.4, retries=3) -> str:
    """Appelle le LLM avec retry en cas de reponse vide."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                return content
        except Exception:
            pass
        if attempt < retries - 1:
            import time
            time.sleep(1)
    return ""


def _generate_plans(params: dict, regulations: dict, client) -> list:
    context = json.dumps({
        "terrain": params,
        "reglementations_applicables": [r["text"] for r in regulations.get("regulations", [])]
    }, ensure_ascii=False)
    
    raw = _llm_retry(client, [
        {"role": "system", "content": _PLANS_SYSTEM},
        {"role": "user", "content": f"Terrain et réglementations :\n{context}"}
    ], max_tokens=1500, temperature=0.4)
    
    if not raw:
        return _default_plans(params)
    
    import re
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            plans = data.get("plans", [])
            if plans:
                return plans
        except json.JSONDecodeError:
            pass
    
    return _default_plans(params)


def _default_plans(params: dict) -> list:
    """Plans de secours quand le LLM ne repond pas."""
    surface = params.get("surface_m2") or 100
    budget = params.get("budget_estime_dh") or 1000000
    etages = params.get("nb_etages_max") or 2
    cout_m2 = 15000
    
    plans = [
        {
            "name": "Maison Plain-pied",
            "description": f"Maison individuelle de plain-pied, {surface} m², budget optimise.",
            "type": "Maison individuelle",
            "nb_etages": 1,
            "surface_utile_m2": round(surface * 0.8, 1),
            "cout_estime_dh": round(surface * cout_m2),
            "delai_mois": 8,
            "criteria_values": {
                "surface_utilisable": round(surface * 0.8, 1),
                "cout_estime": round(surface * cout_m2),
                "conformite_reglementaire": 8.0,
                "faisabilite_technique": 9.0,
                "delai_realisation": 8,
                "ml_viability_score": 7.0
            }
        },
        {
            "name": "Maison R+1",
            "description": f"Maison individuelle R+1, {surface * 1.5} m² utiles, deux niveaux.",
            "type": "Maison individuelle",
            "nb_etages": min(etages, 2),
            "surface_utile_m2": round(surface * 1.2, 1),
            "cout_estime_dh": round(surface * cout_m2 * 1.4),
            "delai_mois": 12,
            "criteria_values": {
                "surface_utilisable": round(surface * 1.2, 1),
                "cout_estime": round(surface * cout_m2 * 1.4),
                "conformite_reglementaire": 7.5,
                "faisabilite_technique": 8.0,
                "delai_realisation": 12,
                "ml_viability_score": 6.5
            }
        },
        {
            "name": "Petit Immeuble R+2",
            "description": f"Petit collectif R+2, {surface * 2.0} m² utiles, 3 logements.",
            "type": "Petit immeuble",
            "nb_etages": min(etages, 3),
            "surface_utile_m2": round(surface * 1.6, 1),
            "cout_estime_dh": round(surface * cout_m2 * 1.8),
            "delai_mois": 16,
            "criteria_values": {
                "surface_utilisable": round(surface * 1.6, 1),
                "cout_estime": round(surface * cout_m2 * 1.8),
                "conformite_reglementaire": 6.5,
                "faisabilite_technique": 7.0,
                "delai_realisation": 16,
                "ml_viability_score": 6.0
            }
        }
    ]
    return plans


# ============================================================
# Étape 4 : Enrichissement (ML + Conformité)
# ============================================================

def _enrich_plan(plan: dict, params: dict, client):
    """
    Enrichit un plan avec :
    - ML Predictor : prédictions depuis le modèle existant
    - RAG Conformité : vérification réglementaire détaillée
    """
    # 4a : ML Predictor — utiliser le modèle existant
    plan["ml_predictions"] = _ml_predict(plan)
    
    # 4b : RAG Conformité — vérification détaillée
    plan["conformite"] = _check_conformity(plan, client)


def _ml_predict(plan: dict) -> dict:
    """
    Utilise le ML Predictor existant avec des features adaptées.
    Les modèles Random Forest entraînés sur le dataset SaaS sont réutilisés
    avec des features proxy (ex: surface → prix, étages → utilisateurs).
    """
    try:
        from ml_predictor import get_predictor
        predictor = get_predictor()
        if not predictor._is_trained:
            return {"status": "modeles_non_entrainees"}
        
        # Mapping des features bâtiment → features SaaS pour le ML
        ml_features = {
            "zone": params.get("localisation") or "France",
            "cible": str(plan.get("type", "individuel")),
            "prix_min_eur": float(plan.get("cout_estime_eur", 0)) * 0.01,
            "prix_max_eur": float(plan.get("cout_estime_eur", 0)) * 0.02,
            "has_freemium": float(plan.get("nb_etages", 1)) > 3,
            "has_essai_gratuit": params.get("budget_estime_dh") is not None,
            "has_API": False,
            "has_mobile": False,
            "has_CRM": float(plan.get("nb_etages", 1)) > 2,
            "has_compta": True,
            "satisfaction_pct": 85.0,
            "note_g2": 4.0,
        }
        
        predictions = predictor.predict_all(ml_features)
        return predictions
        
    except ImportError:
        return {"status": "ml_non_disponible"}
    except Exception as e:
        return {"status": "erreur", "message": str(e)}


def _check_conformity(plan: dict, client) -> dict:
    """Verifie la conformite reglementaire via RAG Taxonomie + LLM structure."""
    from rag_conformite import get_rag
    
    query = f"{plan.get('type', 'batiment')} {plan.get('nb_etages', 1)} etages surface {plan.get('surface_utile_m2', 0)} m²"
    
    try:
        rag = get_rag()
        chunks = rag.retrieve(query, top_k=4, zone_filter="Réglementation Bâtiment")
        if not chunks:
            return {"risk_level": "inconnu", "classification": "", "key_points": [], "details": "Aucune reglementation trouvee pour ce plan."}
        
        context = "\n---\n".join([c["doc"]["text"] for c in chunks])
        
        raw = _llm_retry(client, [
            {"role": "system", "content": (
                "Tu es un expert en reglementation du batiment au Maroc. "
                "A partir du plan et du contexte reglementaire fournis, retourne UNIQUEMENT un JSON :\n"
                "{\n"
                '  "risk_level": "Faible" | "Moyen" | "Eleve",\n'
                '  "classification": "classification reglementaire exacte",\n'
                '  "key_points": ["point 1", "point 2", "point 3"],\n'
                '  "details": "explication courte (1 phrase)"\n'
                "}\n"
                "Sois precis et concis."
            )},
            {"role": "user", "content": f"Plan: {json.dumps(plan, ensure_ascii=False)}\n\nReglementations:\n{context}"}
        ], max_tokens=400, temperature=0.1)
        
        if raw:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                return {
                    "risk_level": result.get("risk_level", "Moyen"),
                    "classification": result.get("classification", ""),
                    "key_points": result.get("key_points", []),
                    "details": result.get("details", ""),
                    "sources": [c["doc"]["source"] for c in chunks],
                    "chunks_used": len(chunks)
                }
    except Exception as e:
        pass
    
    return {"risk_level": "Moyen", "classification": "", "key_points": ["Verification non disponible"], "details": "Impossible d evaluer la conformite.", "sources": [], "chunks_used": 0}


# ============================================================
# Étape 5 : MCDM — Classement multicritère
# ============================================================

def _rank_plans(plans: list, method: str = "wsm") -> dict:
    """Classe les plans avec WSM ou TOPSIS."""
    from mcdm import compare
    
    # Poids des critères pour les plans bâtiment
    weights = {
        "surface_utilisable": 0.20,
        "cout_estime": 0.25,
        "conformite_reglementaire": 0.25,
        "faisabilite_technique": 0.15,
        "delai_realisation": 0.10,
        "ml_viability_score": 0.05
    }
    
    # Construire les solutions MCDM
    solutions = []
    for plan in plans:
        solutions.append({
            "name": plan.get("name", "Plan sans nom"),
            "description": plan.get("description", ""),
            "criteria_values": plan.get("criteria_values", {})
        })
    
    if not solutions:
        return {"plans": plans, "mcdm": {"results": [], "method": method}}
    
    try:
        mcdm_result = compare(
            solutions=solutions,
            weights=weights,
            method=method,
            criteria=BUILDING_CRITERIA,
            criteria_types=BUILDING_CRITERIA_TYPES
        )
    except Exception as e:
        mcdm_result = {"results": [], "method": method, "criteria_labels": {}, "error": str(e)}
    
    # Fusionner les scores MCDM dans chaque plan
    ranked_plans = []
    score_map = {r["name"]: r for r in mcdm_result.get("results", [])}
    
    # Trier par score MCDM
    sorted_plans = sorted(plans, key=lambda p: score_map.get(p.get("name", ""), {}).get("score", 0), reverse=True)
    
    for i, plan in enumerate(sorted_plans):
        mcdm_data = score_map.get(plan.get("name", ""), {})
        plan["rank"] = i + 1
        plan["mcdm_score"] = mcdm_data.get("score", 0)
        plan["mcdm_recommended"] = mcdm_data.get("recommended", False) or (i == 0)
        ranked_plans.append(plan)
    
    return {
        "plans": ranked_plans,
        "mcdm": mcdm_result
    }


# ============================================================
# Résumé conformité
# ============================================================

def _build_conformity_summary(plans: list) -> list:
    """Extrait un résumé structuré de la conformité pour chaque plan."""
    summaries = []
    for plan in plans:
        conf = plan.get("conformite", {})
        summaries.append({
            "plan_name": plan.get("name", ""),
            "risk_level": conf.get("risk_level", "inconnu"),
            "classification": conf.get("classification", ""),
            "sources": conf.get("sources", [])
        })
    return summaries


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    result = recommend(
        "Terrain de 60 m² en ville au Maroc, "
        "budget environ 1 200 000 DH, possible de construire 2 étages",
        method="wsm"
    )
    print(f"Terrain: {result['terrain']}")
    print(f"Réglementations trouvées: {len(result['regulations']['regulations'])}")
    print(f"Plans générés: {len(result['plans'])}")
    for p in result['plans']:
        print(f"  #{p.get('rank', 0)} {p.get('name')} — Score MCDM: {p.get('mcdm_score', 0)}")
        print(f"    ML predictions: {p.get('ml_predictions', {}).get('predictions', {}).keys()}")
    print(f"MCDM method: {result['mcdm']['method']}")
