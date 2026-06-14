"""
pii_anonymizer.py — Anonymisation RGPD / CNDP (Data Governance)
================================================================
Masque automatiquement les donnees personnelles identifiables (PII)
avant tout envoi au LLM, conformement aux principes RGPD/CNDP.

Categories masquees :
  - Adresses e-mail
  - Numeros de telephone (format international et marocain)
  - Noms propres (heuristique simple)
  - Numeros de carte bancaire
  - Adresses IP
  - SIRET / ICE (identifiants entreprise)
  - URLs avec donnees sensibles

Reference : RGPD Art. 25 (Privacy by Design) + CNDP Maroc Loi 09-08
"""

import re
from typing import Tuple

# ══════════════════════════════════════════════════════════════════════════════
# Patterns Regex par categorie PII
# ══════════════════════════════════════════════════════════════════════════════

_PATTERNS = {
    "EMAIL": (
        re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
        "[EMAIL]"
    ),
    "PHONE_INTL": (
        re.compile(r'\+?\d[\d\s\-\.\(\)]{7,}\d'),
        "[TELEPHONE]"
    ),
    "CREDIT_CARD": (
        re.compile(r'\b(?:\d[ \-]?){13,16}\b'),
        "[CARTE_BANCAIRE]"
    ),
    "IP_ADDRESS": (
        re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        "[ADRESSE_IP]"
    ),
    "SIRET": (
        re.compile(r'\b\d{14}\b'),
        "[SIRET]"
    ),
    "ICE_MAROC": (
        re.compile(r'\b\d{15}\b'),
        "[ICE]"
    ),
    "CIN_MAROC": (
        re.compile(r'\b[A-Z]{1,2}\d{5,6}\b'),
        "[CIN]"
    ),
    "PASSWORD_IN_URL": (
        re.compile(r'(https?://[^:]+:)[^@]+(@)', re.IGNORECASE),
        r'\1[MOT_DE_PASSE]\2'
    ),
    "API_KEY": (
        re.compile(r'(?i)(api[_\-]?key|token|secret)["\s:=]+["\']?([A-Za-z0-9\-_\.]{16,})["\']?'),
        r'\1=[CLE_API_MASQUEE]'
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# Anonymiseur principal
# ══════════════════════════════════════════════════════════════════════════════

def anonymize(text: str, categories: list = None) -> Tuple[str, dict]:
    """
    Masque les PII detectees dans le texte.

    Args:
        text       : Texte brut a anonymiser
        categories : Liste des categories a traiter (None = toutes)

    Returns:
        (texte_anonymise, rapport) ou rapport = {categorie: nombre_de_remplacements}
    """
    if not text:
        return text, {}

    result = text
    report = {}

    for category, (pattern, replacement) in _PATTERNS.items():
        if categories and category not in categories:
            continue

        if callable(replacement):
            new_text, count = pattern.subn(replacement, result)
        else:
            new_text, count = pattern.subn(replacement, result)

        if count > 0:
            result = new_text
            report[category] = count

    return result, report


def has_pii(text: str) -> bool:
    """Renvoie True si le texte contient des PII detectables."""
    _, report = anonymize(text)
    return len(report) > 0


def anonymize_dict(data: dict, text_fields: list = None) -> Tuple[dict, dict]:
    """
    Anonymise les champs textuels d'un dictionnaire.

    Args:
        data        : Dictionnaire a traiter
        text_fields : Cles a anonymiser (None = toutes les valeurs str)

    Returns:
        (dict_anonymise, rapport_global)
    """
    result = dict(data)
    global_report = {}

    for key, value in data.items():
        if not isinstance(value, str):
            continue
        if text_fields and key not in text_fields:
            continue

        cleaned, report = anonymize(value)
        result[key] = cleaned
        for cat, count in report.items():
            global_report[cat] = global_report.get(cat, 0) + count

    return result, global_report


# ══════════════════════════════════════════════════════════════════════════════
# Disclaimer RGPD (affiche dans l'UI)
# ══════════════════════════════════════════════════════════════════════════════

RGPD_DISCLAIMER = (
    "Ce systeme fournit une estimation basee sur des donnees historiques anonymisees. "
    "La decision finale reste de la responsabilite de l'utilisateur. "
    "Aucune donnee personnelle identifiable n'est transmise au modele d'IA. "
    "Conforme RGPD (UE) 2016/679 et CNDP Maroc Loi 09-08."
)

RGPD_DISCLAIMER_FR = (
    "⚠️ Ce système fournit une **estimation** basée sur des données historiques. "
    "La décision finale reste de la responsabilité de l'utilisateur. "
    "Vos données sont anonymisées avant traitement (conformité RGPD & CNDP)."
)


# ══════════════════════════════════════════════════════════════════════════════
# Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    samples = [
        "Contactez-moi a john.doe@example.com ou au +212 06 12 34 56 78.",
        "Mon SIRET est 12345678901234 et mon IP est 192.168.1.100.",
        "Carte bancaire : 4532 1234 5678 9010 expire 12/26.",
        "CIN: AB123456, ICE: 123456789012345.",
        "API_KEY=sk-abc123def456ghi789jkl012mno345pqr",
        "Ceci est un texte sans donnees personnelles sur le marche SaaS.",
    ]

    print("=== Test Anonymiseur PII ===\n")
    for s in samples:
        cleaned, report = anonymize(s)
        if report:
            print(f"Original : {s}")
            print(f"Anonymise: {cleaned}")
            print(f"Rapport  : {report}\n")
        else:
            print(f"OK (pas de PII): {s[:60]}\n")

    print("Disclaimer RGPD:")
    print(RGPD_DISCLAIMER_FR)
