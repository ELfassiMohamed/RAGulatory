"""
audit_scraper.py — Mode Audit Web (Analyse en temps reel)
==========================================================
L'utilisateur entre un nom ou une URL de SaaS.
L'agent recupere les informations publiques disponibles,
les normalise, puis les compare avec les modeles ML.

Pipeline :
  1. Recherche web (via Groq LLM + recherche semantique sur le dataset)
  2. Extraction des caracteristiques (prix, fonctionnalites, zone...)
  3. Prediction ML sur ces caracteristiques
  4. Rapport de comparaison : donnees reelles vs predictions

Note : Le vrai web scraping (BeautifulSoup) necessite des packages
externes. Nous utilisons ici le LLM comme moteur d'extraction
et le dataset interne comme reference de comparaison.
"""

import re
import json
import time
import httpx
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME, OPENROUTER_BASE_URL

# ══════════════════════════════════════════════════════════════════════════════
# Prompt d'extraction pour le Mode Audit
# ══════════════════════════════════════════════════════════════════════════════

_AUDIT_EXTRACT_SYSTEM = """Tu es un analyste SaaS expert. L'utilisateur te donne le nom d'un logiciel SaaS.
Extrais les informations suivantes et retourne UNIQUEMENT un JSON valide :

{
  "nom": string,
  "description": string (1 phrase max),
  "secteur": string (ex: "CRM", "ERP", "Facturation", "RH", "Support"),
  "prix_min_eur": float | null,
  "prix_max_eur": float | null,
  "has_freemium": boolean,
  "has_essai_gratuit": boolean,
  "has_API": boolean,
  "has_mobile": boolean,
  "has_CRM": boolean,
  "has_compta": boolean,
  "zone": string ("France" | "Maroc" | "Mondial" | "UE"),
  "cible": string ("PME" | "TPE" | "Grande entreprise" | "Startup" | "Tous"),
  "note_g2": float | null,
  "satisfaction_pct": float | null,
  "certifications": [string],
  "hebergement": string | null,
  "conforme_rgpd": boolean | null,
  "source_info": string (d'ou viennent ces infos : "connaissance generale" | "base de donnees interne")
}

Si tu ne connais pas une valeur, mets null. Ne pas inventer de prix sans base.
Retourne UNIQUEMENT le JSON.
"""

_AUDIT_REPORT_SYSTEM = """Tu es l'Agent Audit SaaS. Tu as analyse un logiciel et obtenu ses caracteristiques
ainsi que les predictions de 4 modeles d'IA. Genere un rapport d'audit structure en FRANCAIS.

Le rapport doit contenir :
1. **Fiche d'identite** — nom, secteur, cible, zone
2. **Analyse tarifaire** — prix vs marche, anomalie detectee ?
3. **Profil fonctionnel** — fonctionnalites presentes / absentes
4. **Predictions IA** — classe de prix, conformite e-facture, satisfaction, segment cible
5. **Verdict de conformite** — RGPD / CNDP
6. **Recommandation finale** — en 2-3 phrases, faut-il acheter ce SaaS ?

Sois precis, factuel et professionnel. Indique toujours le niveau de confiance des predictions.
Termine par la clause : "Ce rapport est genere automatiquement. La decision finale reste de la responsabilite de l'utilisateur."
"""


# ══════════════════════════════════════════════════════════════════════════════
# Classe principale
# ══════════════════════════════════════════════════════════════════════════════

class AuditScraper:
    """
    Agent Mode Audit : extrait + analyse + rapport pour un SaaS donne.
    """

    def __init__(self):
        http_client = httpx.Client(trust_env=False)
        self.client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, http_client=http_client)

    def extract_features(self, saas_name_or_url: str) -> dict:
        """
        Etape 1 : Extraction des caracteristiques du SaaS a partir de son nom/URL.
        Utilise le LLM comme moteur de connaissance et d'extraction.
        """
        # Nettoyer l'entree
        query = saas_name_or_url.strip()
        if query.startswith("http"):
            # Extraire le nom de domaine
            domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/\s]+)', query)
            if domain_match:
                query = domain_match.group(1).split('.')[0].capitalize()

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": _AUDIT_EXTRACT_SYSTEM},
                    {"role": "user", "content": f"Analyse ce logiciel SaaS : {query}"}
                ],
                temperature=0.1,
                max_tokens=600
            )

            raw = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                features = json.loads(json_match.group(0))
                features["_query"] = query
                return features

        except Exception as e:
            pass

        # Fallback minimal
        return {
            "nom": query,
            "description": "Informations non disponibles",
            "_query": query,
            "_error": "Extraction echouee"
        }

    def run_predictions(self, features: dict) -> dict:
        """
        Etape 2 : Lance les 4 predictions ML sur les features extraites.
        """
        try:
            from ml_predictor import get_predictor
            predictor = get_predictor()
            if predictor._is_trained:
                return predictor.predict_all(features)
        except Exception:
            pass
        return {"predictions": {}, "explanation": [], "anomaly": {}}

    def compare_with_dataset(self, features: dict) -> dict:
        """
        Etape 3 : Compare le SaaS audite avec le dataset interne.
        Retourne les meilleurs concurrents similaires.
        """
        try:
            from dataset_loader import load_dataset, clean_dataset
            import pandas as pd

            df = clean_dataset(load_dataset())

            # Filtrer par secteur/cible similaire
            zone = features.get("zone", "Mondial")
            cible = features.get("cible", "PME")

            similar = df[
                (df["zone"].str.lower() == zone.lower()) |
                (df["cible"].str.lower() == cible.lower())
            ]

            if similar.empty:
                similar = df

            # Statistiques de comparaison
            prix_max = features.get("prix_max_eur") or 0
            market_median = float(similar["prix_max_eur"].median())
            market_mean   = float(similar["prix_max_eur"].mean())
            market_q75    = float(similar["prix_max_eur"].quantile(0.75))

            ecart_pct = ((prix_max - market_median) / market_median * 100) if market_median > 0 and prix_max > 0 else None

            return {
                "comparaison_marche": {
                    "n_similaires":    len(similar),
                    "prix_median_eur": round(market_median, 2),
                    "prix_moyen_eur":  round(market_mean, 2),
                    "prix_q75_eur":    round(market_q75, 2),
                    "ecart_pct":       round(ecart_pct, 1) if ecart_pct is not None else None,
                    "satisfaction_moy": round(float(similar["satisfaction_pct"].mean()), 1),
                    "pct_avec_api":    round(float(similar["has_API"].mean()) * 100, 0),
                    "pct_freemium":    round(float(similar["a_freemium"].mean()) * 100, 0),
                }
            }
        except Exception as e:
            return {"comparaison_marche": {"error": str(e)}}

    def generate_report(self, saas_name: str, features: dict, predictions: dict, comparison: dict, lang: str = "fr") -> str:
        """
        Etape 4 : Genere le rapport d'audit complet via le LLM.
        """
        lang_rule = "Reponds UNIQUEMENT en FRANCAIS." if lang == "fr" else "Respond ONLY in ENGLISH."
        context = json.dumps({
            "saas": features,
            "predictions_ia": {
                k: {"label": v.get("label"), "confidence": v.get("confidence"), "accuracy": v.get("accuracy")}
                for k, v in (predictions.get("predictions") or {}).items()
            },
            "anomalie_prix": predictions.get("anomaly", {}),
            "comparaison_marche": comparison.get("comparaison_marche", {})
        }, ensure_ascii=False, indent=2)

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": _AUDIT_REPORT_SYSTEM + "\n" + lang_rule},
                    {"role": "user", "content": f"Voici les donnees d'audit :\n{context}"}
                ],
                temperature=0.3,
                max_tokens=900
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erreur de generation du rapport : {e}"

    def full_audit(self, saas_name_or_url: str, lang: str = "fr") -> dict:
        """
        Pipeline complet : extraction → predictions → comparaison → rapport.
        """
        t0 = time.time()

        # 1. Extraction
        features = self.extract_features(saas_name_or_url)

        # 2. Predictions ML
        predictions = self.run_predictions(features)

        # 3. Comparaison dataset
        comparison = self.compare_with_dataset(features)

        # 4. Rapport
        report = self.generate_report(
            saas_name_or_url, features, predictions, comparison, lang
        )

        duration_ms = (time.time() - t0) * 1000

        # Log
        try:
            from logger import log_audit
            log_audit(
                saas_name=features.get("nom", saas_name_or_url),
                url=saas_name_or_url if saas_name_or_url.startswith("http") else "",
                features_scraped=features,
                predictions=predictions.get("predictions", {}),
                duration_ms=duration_ms
            )
        except Exception:
            pass

        return {
            "saas_name":  features.get("nom", saas_name_or_url),
            "features":   features,
            "predictions": predictions,
            "comparison": comparison,
            "report":     report,
            "duration_ms": round(duration_ms, 0)
        }


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_scraper_instance = None

def get_scraper() -> AuditScraper:
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = AuditScraper()
    return _scraper_instance


if __name__ == "__main__":
    scraper = AuditScraper()
    print("Test extraction features pour 'Salesforce'...")
    features = scraper.extract_features("Salesforce")
    print(json.dumps(features, ensure_ascii=False, indent=2))
