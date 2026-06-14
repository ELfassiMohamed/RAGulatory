"""
nlu_extractor.py — Module NLU/NER (Natural Language Understanding)
Extrait des critères structurés depuis du texte libre en utilisant le LLM Groq
comme moteur Transformer, sans dépendance externe lourde.

Architecture : Texte Libre → Prompt NER structuré → Groq API → JSON criteria
"""

import json
import re
import httpx
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME, OPENROUTER_BASE_URL

# ============================================================
# Prompt système NER
# ============================================================

_NLU_SYSTEM_PROMPT = """Tu es un moteur NLU (Natural Language Understanding) spécialisé dans l'extraction d'entités métiers pour l'achat de logiciels SaaS.

Analyse le texte de l'utilisateur et extrait les entités suivantes. Retourne UNIQUEMENT un objet JSON valide, sans texte avant ni après.

Schéma JSON attendu :
{
  "secteur": string | null,           // Secteur métier (ex: "Comptabilité", "RH", "CRM", "ERP", "Support client")
  "utilisateurs": integer | null,     // Nombre d'utilisateurs/employés mentionné
  "zone": string | null,              // Zone géographique (ex: "Maroc", "France", "UE", "Mondial")
  "budget_max_dh": float | null,      // Budget maximum en Dirhams (DH)
  "budget_max_eur": float | null,     // Budget maximum en Euros (EUR)
  "duree_mois": integer | null,       // Durée d'engagement souhaitée en mois
  "integrationsRequises": [string],   // Intégrations mentionnées (ex: ["Salesforce", "Slack", "ERP"])
  "fonctionnalites": [string],        // Fonctionnalités demandées (ex: ["facturation", "API", "mobile"])
  "contraintes_securite": boolean,    // Mention explicite de sécurité / RGPD / CNDP
  "has_freemium": boolean,            // Mentionne un plan gratuit / freemium
  "priorite": string | null           // Priorité principale exprimée ("prix", "securite", "fonctionnalites", "facilite")
}

Règles :
- Si une valeur n'est pas mentionnée, mettre null ou [] selon le type.
- Convertir les montants : 1 EUR ≈ 10.8 DH. Si budget en DH → calculer EUR. Si EUR → calculer DH.
- Retourne UNIQUEMENT du JSON valide.
"""

# ============================================================
# Extracteur principal
# ============================================================

class NLUExtractor:
    """
    Extrait des critères structurés depuis du texte libre.
    Utilise le LLM comme pipeline Transformer NER/NLU.
    """

    def __init__(self):
        http_client = httpx.Client(trust_env=False)
        self.client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, http_client=http_client)

    def extract(self, user_text: str) -> dict:
        """
        Pipeline principal : Texte libre → critères JSON structurés.

        Args:
            user_text: Description libre de l'utilisateur

        Returns:
            dict avec les entités extraites + score de confiance
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": _NLU_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Texte à analyser : {user_text}"}
                ],
                temperature=0.0,   # Déterministe pour l'extraction
                max_tokens=500
            )

            raw = response.choices[0].message.content.strip()

            # Extraire le JSON si entouré de texte parasite
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                raw = json_match.group(0)

            criteria = json.loads(raw)
            criteria["_confidence"] = self._compute_confidence(criteria)
            criteria["_source_text"] = user_text
            return criteria

        except (json.JSONDecodeError, Exception) as e:
            # Fallback : retourner un dict vide avec erreur
            return {
                "secteur": None,
                "utilisateurs": None,
                "zone": None,
                "budget_max_dh": None,
                "budget_max_eur": None,
                "duree_mois": None,
                "integrationsRequises": [],
                "fonctionnalites": [],
                "contraintes_securite": False,
                "has_freemium": False,
                "priorite": None,
                "_confidence": 0,
                "_error": str(e),
                "_source_text": user_text
            }

    def _compute_confidence(self, criteria: dict) -> int:
        """
        Score de confiance de l'extraction (0-100).
        Basé sur le nombre de champs renseignés.
        """
        key_fields = ["secteur", "utilisateurs", "zone", "budget_max_eur", "budget_max_dh"]
        secondary_fields = ["fonctionnalites", "integrationsRequises", "duree_mois", "priorite"]

        score = 0
        for f in key_fields:
            if criteria.get(f) not in (None, [], ""):
                score += 16
        for f in secondary_fields:
            val = criteria.get(f)
            if val not in (None, [], ""):
                score += 5
        return min(score, 100)

    def to_mcdm_weights(self, criteria: dict) -> dict:
        """
        Convertit les critères extraits en poids MCDM suggérés.
        Implémente la logique neuro-symbolique : NLU → MCDM.
        """
        weights = {"fonctionnalite": 0.30, "cout_tco": 0.40, "securite_sla": 0.30}

        # Ajuster selon la priorité exprimée
        priorite = criteria.get("priorite")
        if priorite == "prix":
            weights = {"fonctionnalite": 0.25, "cout_tco": 0.55, "securite_sla": 0.20}
        elif priorite == "securite":
            weights = {"fonctionnalite": 0.20, "cout_tco": 0.30, "securite_sla": 0.50}
        elif priorite == "fonctionnalites":
            weights = {"fonctionnalite": 0.55, "cout_tco": 0.25, "securite_sla": 0.20}
        elif priorite == "facilite":
            weights = {"fonctionnalite": 0.45, "cout_tco": 0.35, "securite_sla": 0.20}

        # Renforcer sécurité si RGPD/CNDP mentionné
        if criteria.get("contraintes_securite"):
            weights["securite_sla"] = min(weights["securite_sla"] + 0.15, 0.60)
            total = sum(weights.values())
            weights = {k: round(v / total, 3) for k, v in weights.items()}

        return weights

    def to_roi_params(self, criteria: dict) -> dict:
        """
        Convertit les critères extraits en paramètres ROI/TCO suggérés.
        """
        params = {}
        if criteria.get("utilisateurs"):
            params["n_utilisateurs"] = criteria["utilisateurs"]
        if criteria.get("budget_max_eur"):
            params["prix_mensuel"] = round(criteria["budget_max_eur"] / max(criteria.get("utilisateurs") or 1, 1), 2)
        if criteria.get("duree_mois"):
            params["duree_mois"] = criteria["duree_mois"]
        return params


# ============================================================
# Singleton
# ============================================================

_extractor_instance = None

def get_extractor() -> NLUExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = NLUExtractor()
    return _extractor_instance


def extract_criteria(text: str) -> dict:
    """Point d'entrée simplifié."""
    return get_extractor().extract(text)


# ============================================================
# Test rapide
# ============================================================

if __name__ == "__main__":
    extractor = NLUExtractor()
    prompts = [
        "Je cherche un SaaS de comptabilité pour 15 employés au Maroc, budget max 3000 DH/mois.",
        "On a besoin d'un ERP cloud sécurisé pour notre PME de 50 personnes en France, conformité RGPD obligatoire, max 80 EUR/user/mois.",
        "Looking for a CRM with mobile app and Salesforce integration, around 500 users, security is priority."
    ]

    for p in prompts:
        print(f"\nTexte : {p[:60]}…")
        result = extractor.extract(p)
        print(f"  Secteur: {result.get('secteur')} | Users: {result.get('utilisateurs')} | Zone: {result.get('zone')}")
        print(f"  Budget: {result.get('budget_max_eur')} EUR / {result.get('budget_max_dh')} DH")
        print(f"  Confiance: {result.get('_confidence')}%")
        weights = extractor.to_mcdm_weights(result)
        print(f"  Poids MCDM suggérés: {weights}")
