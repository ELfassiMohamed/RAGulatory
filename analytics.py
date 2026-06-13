# analytics.py — Modules strategiques avances
import pandas as pd
import os

# Charger Feature_Importance une seule fois
_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_final.xlsx")

def _load_feature_importance():
    try:
        fi = pd.read_excel(_DATASET_PATH, sheet_name="Feature_Importance")
        return fi
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance", "Importance_pct"])

FEATURE_IMPORTANCE = _load_feature_importance()

# ============================================================
# 1. Indicateur de confiance statistique
# ============================================================

def calculer_confiance(segment_df):
    if segment_df is None or len(segment_df) == 0:
        return None
    n = len(segment_df)
    n_real = (segment_df["source"] == "real").sum() if "source" in segment_df.columns else 0
    n_ctgan = (segment_df["source"] == "ctgan").sum() if "source" in segment_df.columns else 0
    n_copula = (segment_df["source"] == "gaussian_copula").sum() if "source" in segment_df.columns else 0
    pct_real = n_real / n * 100 if n > 0 else 0
    if pct_real >= 50:
        niveau = "Eleve"
        score = int(50 + pct_real * 0.5)
    elif pct_real >= 20:
        niveau = "Moyen"
        score = int(30 + pct_real * 1.5)
    else:
        niveau = "Faible (donnees synthetiques predominantes)"
        score = max(10, int(pct_real * 2))
    return {
        "niveau": niveau,
        "score": min(score, 99),
        "n_total": n,
        "n_real": int(n_real),
        "n_synthetique": int(n_ctgan + n_copula),
        "pct_real": round(pct_real, 1)
    }

# ============================================================
# 2. Analyse de segment
# ============================================================

def analyser_segment(df, zone=None, cible=None):
    sub = df.copy()
    if zone and zone.lower() not in ["tous", "all", "mondial"]:
        exact = sub[sub["zone"].str.lower() == zone.lower()]
        sub = exact if not exact.empty else sub[sub["zone"].str.lower().str.contains(zone.lower(), na=False)]
    if cible and cible.lower() not in ["tous", "all"]:
        exact = sub[sub["cible"].str.lower() == cible.lower()]
        sub = exact if not exact.empty else sub[sub["cible"].str.lower().str.contains(cible.lower(), na=False)]
    if sub.empty:
        return None

    n = len(sub)
    conf = calculer_confiance(sub)
    conf_str = f"Indice de confiance : {conf['niveau']} ({conf['score']}%) — {conf['n_real']} observations reelles sur {conf['n_total']}" if conf else ""

    insights = [
        f"Concurrents identifies sur ce segment : {n}",
        f"Prix minimum median : {sub['prix_min_eur'].median():.2f} EUR",
        f"Prix maximum median : {sub['prix_max_eur'].median():.2f} EUR",
        f"Adoption freemium : {sub['a_freemium'].mean()*100:.0f}% des concurrents",
        f"Essai gratuit : {sub['a_essai_gratuit'].mean()*100:.0f}% des concurrents",
        f"Tarification a l'usage : {sub['a_usage_based'].mean()*100:.0f}% des concurrents",
        f"Satisfaction moyenne : {sub['satisfaction_pct'].mean():.1f}%",
        f"Note G2 moyenne : {sub['note_g2'].mean():.2f}/5",
    ]
    if conf_str:
        insights.append(conf_str)

    features = {"has_API": "API", "has_CRM": "CRM integre", "has_compta": "Comptabilite",
                "has_mobile": "App mobile", "has_PDP": "PDP", "conforme_e_facture_2026": "e-facture 2026"}
    must_have, differentiators = [], []
    for col, label in features.items():
        if col in sub.columns:
            rate = sub[col].mean()
            if rate >= 0.70:
                must_have.append(f"{label} ({rate*100:.0f}%)")
            elif rate <= 0.30:
                differentiators.append(f"{label} ({rate*100:.0f}% seulement)")
    if must_have:
        insights.append(f"Fonctionnalites INDISPENSABLES (>70%) : {', '.join(must_have)}")
    if differentiators:
        insights.append(f"OPPORTUNITES de differenciation (<30%) : {', '.join(differentiators)}")

    funded = sub[sub["total_financement_mUSD"] > 0]
    if len(funded) > 0:
        insights.append(f"Acteurs finances : {len(funded)}/{n} ({len(funded)/n*100:.0f}%), total {funded['total_financement_mUSD'].sum():.1f}M USD")
    else:
        insights.append("Marche majoritairement bootstrappe (aucun financement majeur detecte)")

    return insights

# ============================================================
# 3. Score de viabilite (simulateur de positionnement)
# ============================================================

def scorer_viabilite(df, prix_envisage, zone=None, cible=None, has_freemium=False, has_usage_based=False):
    sub = df.copy()
    if zone:
        exact = sub[sub["zone"].str.lower() == zone.lower()]
        if not exact.empty:
            sub = exact
    if cible:
        exact = sub[sub["cible"].str.lower() == cible.lower()]
        if not exact.empty:
            sub = exact
    if sub.empty:
        return [], 5

    median_prix = sub["prix_min_eur"].median()
    p90 = sub["prix_min_eur"].quantile(0.90)
    p10 = sub["prix_min_eur"].quantile(0.10)
    taux_freemium = sub["a_freemium"].mean()
    taux_usage = sub["a_usage_based"].mean()

    score = 10
    alertes = []

    if prix_envisage > p90:
        alertes.append(f"ALERTE PRIX : {prix_envisage}EUR depasse le 90e centile ({p90:.0f}EUR). Forte friction attendue sans valeur premium claire. -3 points.")
        score -= 3
    elif prix_envisage > median_prix * 1.5:
        alertes.append(f"AVERTISSEMENT PRIX : {prix_envisage}EUR est 50% au-dessus de la mediane ({median_prix:.0f}EUR). Justifier le ROI. -1 point.")
        score -= 1
    elif prix_envisage < p10:
        alertes.append(f"PRIX TRES BAS : {prix_envisage}EUR est sous le 10e centile. Risque de positionnement 'low-cost' difficile a valoriser. -1 point.")
        score -= 1
    else:
        alertes.append(f"PRIX COHERENT : {prix_envisage}EUR est dans la norme du segment (mediane {median_prix:.0f}EUR). +0")

    if not has_freemium and taux_freemium > 0.5:
        alertes.append(f"RISQUE ACQUISITION : {taux_freemium*100:.0f}% des concurrents offrent un freemium. Sans essai gratuit, CAC plus eleve. -1 point.")
        score -= 1
    elif has_freemium:
        alertes.append(f"BONNE PRATIQUE : Freemium present — aligne avec {taux_freemium*100:.0f}% du marche. +1 point.")
        score += 1

    if has_usage_based and taux_usage > 0.3:
        alertes.append(f"AVANTAGE STRATEGIQUE : Tarification a l'usage adoptee — facteur d'impact N°1 selon notre modele (15.39% Feature Importance). +2 points.")
        score += 2
    elif not has_usage_based and taux_usage > 0.5:
        alertes.append(f"OPPORTUNITE MANQUEE : {taux_usage*100:.0f}% du segment utilise la tarification a l'usage (facteur le plus impactant). -1 point.")
        score -= 1

    score = max(1, min(10, score))
    alertes.append(f"SCORE DE VIABILITE : {score}/10")
    return alertes, score

# ============================================================
# 4. Detecteur d'oceans bleus
# ============================================================

def detecter_ocean_bleu(df):
    resultats = []
    zones = df["zone"].unique()
    cibles = df["cible"].unique()
    for zone in zones:
        for cible in cibles:
            sub = df[(df["zone"] == zone) & (df["cible"] == cible)]
            if len(sub) < 5:
                continue
            sat = sub["satisfaction_pct"].mean()
            n = len(sub)
            # Ocean bleu : peu de concurrents OU insatisfaction elevee
            if n <= 15 and sat < 75:
                mobile_gap = sub["has_mobile"].mean() < 0.3 if "has_mobile" in sub.columns else False
                api_gap = sub["has_API"].mean() < 0.4 if "has_API" in sub.columns else False
                opp = []
                if mobile_gap:
                    opp.append("app mobile absente")
                if api_gap:
                    opp.append("API rare")
                resultats.append({
                    "zone": zone,
                    "cible": cible,
                    "n_concurrents": n,
                    "satisfaction": round(sat, 1),
                    "opportunites": opp
                })
    resultats.sort(key=lambda x: (x["n_concurrents"], x["satisfaction"]))
    return resultats[:5]

# ============================================================
# 5. Feature Importance - recommandations MVP
# ============================================================

def recommandations_mvp():
    if FEATURE_IMPORTANCE.empty:
        return []
    top = FEATURE_IMPORTANCE.head(5)
    recs = []
    mapping = {
        "a_usage_based": "Tarification a l'usage (Usage-Based Pricing)",
        "prix_max_eur": "Gamme de prix maximum competitive",
        "note_moyenne_globale": "Excellence de la satisfaction client (notes G2/Capterra)",
        "nb_avis_g2": "Strategie d'acquisition d'avis G2",
        "nb_plans": "Diversite des plans tarifaires",
        "has_API": "API disponible",
        "a_freemium": "Modele Freemium",
        "a_essai_gratuit": "Essai gratuit",
    }
    for _, row in top.iterrows():
        feat = row["Feature"]
        pct = row["Importance_pct"]
        label = mapping.get(feat, feat)
        recs.append(f"{label} ({pct}% d'importance predictive)")
    return recs

# ============================================================
# 6. Conformite reglementaire France 2026
# ============================================================

def audit_conformite(df, zone=None, cible=None):
    if zone and "france" not in zone.lower() and "europe" not in zone.lower():
        return []
    sub = df.copy()
    if zone:
        exact = sub[sub["zone"].str.lower() == zone.lower()]
        sub = exact if not exact.empty else sub
    if cible:
        exact = sub[sub["cible"].str.lower() == cible.lower()]
        sub = exact if not exact.empty else sub
    if sub.empty:
        return []
    pct_conforme = sub["conforme_e_facture_2026"].mean() * 100 if "conforme_e_facture_2026" in sub.columns else 0
    pct_pdp = sub["has_PDP"].mean() * 100 if "has_PDP" in sub.columns else 0
    return [
        f"ALERTE REGLEMENTAIRE France 2026 : La facturation electronique obligatoire entre en vigueur.",
        f"Sur votre segment : {pct_conforme:.0f}% des concurrents sont deja conformes e-facture 2026.",
        f"Connexion PDP (Plateforme de Dematerialisation Partenaire) : {pct_pdp:.0f}% des acteurs.",
        "RECOMMANDATION : Integrer la conformite Facture-X et la connectivite PDP/PPF des le MVP si vous ciblez la France.",
    ]

# ============================================================
# 7. Detection de segment dans le texte
# ============================================================

def detecter_segment(question):
    q = question.lower()
    zone = None
    for z in ["france", "maroc", "europe", "mondial", "usa", "afrique"]:
        if z in q:
            zone = z.capitalize()
            break
    cible = None
    for c in ["tpe", "pme", "eti", "startup", "grande entreprise", "freelance"]:
        if c in q:
            cible = c.upper() if c in ["tpe", "pme", "eti"] else c.capitalize()
            break
    return zone, cible

# ============================================================
# 8. Generateur de rapport Pitch Deck
# ============================================================

def generer_pitch_deck(df, zone=None, cible=None, prix=None, has_freemium=False, has_usage=False):
    sub = df.copy()
    if zone:
        exact = sub[sub["zone"].str.lower() == zone.lower()]
        sub = exact if not exact.empty else sub
    if cible:
        exact = sub[sub["cible"].str.lower() == cible.lower()]
        sub = exact if not exact.empty else sub

    n = len(sub)
    conf = calculer_confiance(sub)
    fi_recs = recommandations_mvp()
    features = {"has_API": "API", "has_CRM": "CRM", "has_compta": "Comptabilite",
                "has_mobile": "App mobile", "conforme_e_facture_2026": "e-facture 2026"}
    must_have = [label for col, label in features.items() if col in sub.columns and sub[col].mean() >= 0.70]
    nice_to_have = [label for col, label in features.items() if col in sub.columns and 0.30 <= sub[col].mean() < 0.70]

    score_lines = []
    if prix:
        alertes, score = scorer_viabilite(df, prix, zone, cible, has_freemium, has_usage)
        score_lines = alertes

    lines = [
        f"# Pitch Deck — Diagnostic SaaS",
        f"**Segment analysé :** {zone or 'Global'} — {cible or 'Toutes cibles'}",
        f"**Données :** {n} solutions analysées | Confiance : {conf['niveau']} ({conf['score']}%)" if conf else f"**Données :** {n} solutions analysées",
        "",
        "## 1. Market Size & Competitive Landscape",
        f"- Concurrents directs identifiés : **{n}**",
        f"- Prix minimum médian : **{sub['prix_min_eur'].median():.2f} EUR/mois**",
        f"- Prix maximum médian : **{sub['prix_max_eur'].median():.2f} EUR/mois**",
        f"- Freemium adopté par : **{sub['a_freemium'].mean()*100:.0f}%** des acteurs",
        f"- Satisfaction moyenne du marché : **{sub['satisfaction_pct'].mean():.1f}%**",
        "",
        "## 2. Pricing Strategy",
    ]
    if score_lines:
        lines += [f"- {l}" for l in score_lines]
    else:
        lines += [
            f"- Médiane marché : {sub['prix_min_eur'].median():.2f} EUR",
            f"- P10 : {sub['prix_min_eur'].quantile(0.10):.2f} EUR | P90 : {sub['prix_min_eur'].quantile(0.90):.2f} EUR",
        ]
    lines += [
        "",
        "## 3. Product Roadmap — MVP",
        "**Must-Have (>70% des concurrents) :**",
    ]
    for f in must_have:
        lines.append(f"- {f}")
    if nice_to_have:
        lines.append("")
        lines.append("**Nice-to-Have (30-70%) :**")
        for f in nice_to_have:
            lines.append(f"- {f}")
    lines += [
        "",
        "## 4. Facteurs Clés de Succès (Feature Importance)",
    ]
    for i, r in enumerate(fi_recs, 1):
        lines.append(f"{i}. {r}")
    lines += ["", "---", "*Généré par SaaS Assistant*"]
    return "\n".join(lines)

# ============================================================
# 9. Benchmark automatique des concurrents
# ============================================================

def benchmark_concurrents(df, zone=None, cible=None, prix=None,
                           has_mobile=False, has_api=False, has_freemium=False):
    sub = df.copy()
    if zone:
        exact = sub[sub["zone"].str.lower() == zone.lower()]
        sub = exact if not exact.empty else sub
    if cible:
        exact = sub[sub["cible"].str.lower() == cible.lower()]
        sub = exact if not exact.empty else sub
    if sub.empty:
        return []

    prix_marche = sub["prix_min_eur"].median()
    mobile_pct = sub["has_mobile"].mean() * 100 if "has_mobile" in sub.columns else 0
    api_pct = sub["has_API"].mean() * 100 if "has_API" in sub.columns else 0
    freemium_pct = sub["a_freemium"].mean() * 100 if "a_freemium" in sub.columns else 0
    usage_pct = sub["a_usage_based"].mean() * 100 if "a_usage_based" in sub.columns else 0
    sat = sub["satisfaction_pct"].mean()

    rows = [
        ("Prix minimum", f"{prix}EUR" if prix else "N/A", f"{prix_marche:.2f}EUR (mediane)"),
        ("App mobile", "Oui" if has_mobile else "Non", f"{mobile_pct:.0f}% du marche"),
        ("API", "Oui" if has_api else "Non", f"{api_pct:.0f}% du marche"),
        ("Freemium", "Oui" if has_freemium else "Non", f"{freemium_pct:.0f}% du marche"),
        ("Usage-based", "Non", f"{usage_pct:.0f}% du marche"),
        ("Satisfaction marche", "-", f"{sat:.1f}%"),
    ]

    # Score d'alignement
    score_items = []
    if prix and abs(prix - prix_marche) / max(prix_marche, 1) < 0.5:
        score_items.append(1)
    else:
        score_items.append(0)
    if has_mobile or mobile_pct < 50:
        score_items.append(1)
    else:
        score_items.append(0)
    if has_api or api_pct < 50:
        score_items.append(1)
    else:
        score_items.append(0)
    if (has_freemium and freemium_pct > 40) or (not has_freemium and freemium_pct < 40):
        score_items.append(1)
    else:
        score_items.append(0)

    alignment = int(sum(score_items) / len(score_items) * 100)
    return {"rows": rows, "alignment": alignment, "n": len(sub)}

# ============================================================
# 10. Score global de viabilite multi-criteres
# ============================================================

def score_global_viabilite(df, zone=None, cible=None, prix=None,
                             has_mobile=False, has_api=False, has_freemium=False,
                             has_usage=False, target_france=False):
    sub = df.copy()
    if zone:
        exact = sub[sub["zone"].str.lower() == zone.lower()]
        if not exact.empty:
            sub = exact
    if cible:
        exact = sub[sub["cible"].str.lower() == cible.lower()]
        if not exact.empty:
            sub = exact

    scores = {}

    # Market Fit (0-10)
    n = len(sub)
    sat = sub["satisfaction_pct"].mean() if not sub.empty else 70
    scores["Market Fit"] = min(10, round(n / 30 + sat / 20, 1))

    # Pricing (0-10)
    if prix and not sub.empty:
        median_p = sub["prix_min_eur"].median()
        p10 = sub["prix_min_eur"].quantile(0.10)
        p90 = sub["prix_min_eur"].quantile(0.90)
        if p10 <= prix <= p90:
            scores["Pricing"] = 8.0
        elif prix < p10:
            scores["Pricing"] = 6.0
        else:
            scores["Pricing"] = max(3.0, 10 - (prix - p90) / max(median_p, 1) * 3)
        scores["Pricing"] = round(scores["Pricing"], 1)
    else:
        scores["Pricing"] = 5.0

    # Innovation (0-10): usage-based + mobile + API
    innov = 5.0
    if has_usage:
        innov += 2.5
    if has_api and sub.get("has_API", sub["has_API"] if "has_API" in sub.columns else pd.Series([0])).mean() < 0.7:
        innov += 1.0
    if has_mobile and sub.get("has_mobile", sub["has_mobile"] if "has_mobile" in sub.columns else pd.Series([0])).mean() < 0.5:
        innov += 1.5
    scores["Innovation"] = round(min(10, innov), 1)

    # Conformite (0-10)
    if target_france or (zone and "france" in zone.lower()):
        scores["Conformite"] = 10.0  # max si respecte
    else:
        scores["Conformite"] = 8.0

    # Scalabilite (0-10)
    scale = 6.0
    if has_api:
        scale += 2.0
    if has_usage:
        scale += 1.5
    scores["Scalabilite"] = round(min(10, scale), 1)

    final = round(sum(scores.values()) / len(scores), 1)
    return scores, final

# ============================================================
# 11. Analyse SWOT automatique
# ============================================================

def analyse_swot(df, zone=None, cible=None, prix=None,
                  has_mobile=False, has_api=False, has_freemium=False, has_usage=False):
    sub = df.copy()
    if zone:
        exact = sub[sub["zone"].str.lower() == zone.lower()]
        if not exact.empty:
            sub = exact
    if cible:
        exact = sub[sub["cible"].str.lower() == cible.lower()]
        if not exact.empty:
            sub = exact

    median_prix = sub["prix_min_eur"].median() if not sub.empty else 20
    mobile_pct = sub["has_mobile"].mean() * 100 if "has_mobile" in sub.columns and not sub.empty else 50
    api_pct = sub["has_API"].mean() * 100 if "has_API" in sub.columns and not sub.empty else 50
    freemium_pct = sub["a_freemium"].mean() * 100 if "a_freemium" in sub.columns and not sub.empty else 50
    usage_pct = sub["a_usage_based"].mean() * 100 if "a_usage_based" in sub.columns and not sub.empty else 30
    sat = sub["satisfaction_pct"].mean() if not sub.empty else 75

    forces, faiblesses, opportunites, menaces = [], [], [], []

    # Forces
    if prix and prix <= median_prix * 1.1:
        forces.append(f"Prix competitif ({prix}EUR vs mediane {median_prix:.0f}EUR)")
    if has_api:
        forces.append("API disponible — integrabilite elevee")
    if has_freemium:
        forces.append("Modele freemium — barriere a l'entree reduite")
    if has_usage:
        forces.append("Tarification a l'usage — facteur d'impact N°1 (15.39%)")
    if has_mobile:
        forces.append("Application mobile disponible")
    if not forces:
        forces.append("Entree sur un marche avec des donnees disponibles")

    # Faiblesses
    if not has_mobile and mobile_pct > 60:
        faiblesses.append(f"Pas d'app mobile ({mobile_pct:.0f}% des concurrents en ont une)")
    if not has_api and api_pct > 60:
        faiblesses.append(f"Pas d'API ({api_pct:.0f}% des concurrents en ont une)")
    if not has_usage and usage_pct > 35:
        faiblesses.append(f"Absence de tarification a l'usage ({usage_pct:.0f}% du marche)")
    if not has_freemium and freemium_pct > 55:
        faiblesses.append(f"Pas de freemium ({freemium_pct:.0f}% des concurrents en proposent)")
    if not faiblesses:
        faiblesses.append("Pas de faiblesse majeure identifiee sur ce segment")

    # Opportunites
    if not sub.empty and sat < 78:
        opportunites.append(f"Insatisfaction du marche ({sat:.0f}%) — opportunite de disruption")
    if len(sub) < 20 and not sub.empty:
        opportunites.append(f"Marche peu concurrentiel ({len(sub)} acteurs identifies)")
    if not has_usage:
        opportunites.append("Adoption de la tarification a l'usage comme differenciateur fort")
    if zone and "maroc" in zone.lower():
        opportunites.append("Marche emergent avec peu d'acteurs locaux specialises")

    # Menaces
    n = len(sub)
    if n > 50:
        menaces.append(f"Marche sature : {n} concurrents identifies sur ce segment")
    if not has_api and api_pct > 70:
        menaces.append("L'API est un standard du marche — risque d'exclusion technique")
    menaces.append("Evolution rapide des reglementations (e-facture 2026)")

    return {"Forces": forces, "Faiblesses": faiblesses,
            "Opportunites": opportunites, "Menaces": menaces}

# ============================================================
# 12. Simulateur de revenus MRR / ARR
# ============================================================

def simulateur_revenus(prix_mensuel, nb_clients, churn_rate=5.0, cac=150):
    mrr = prix_mensuel * nb_clients
    arr = mrr * 12
    ltv = (prix_mensuel / (churn_rate / 100)) if churn_rate > 0 else prix_mensuel * 24
    ltv_cac = ltv / cac if cac > 0 else 0
    payback = cac / prix_mensuel if prix_mensuel > 0 else 0
    return {
        "mrr": round(mrr, 2),
        "arr": round(arr, 2),
        "ltv": round(ltv, 2),
        "ltv_cac": round(ltv_cac, 2),
        "payback_mois": round(payback, 1),
        "churn_rate": churn_rate
    }

# ============================================================
# 13. Business Model Canvas automatique
# ============================================================

def business_model_canvas(df, zone=None, cible=None, prix=None, has_usage=False):
    sub = df.copy()
    if zone:
        exact = sub[sub["zone"].str.lower() == zone.lower()]
        if not exact.empty:
            sub = exact
    if cible:
        exact = sub[sub["cible"].str.lower() == cible.lower()]
        if not exact.empty:
            sub = exact

    median_prix = sub["prix_min_eur"].median() if not sub.empty else 20
    fi = recommandations_mvp()[:3]

    canvas = {
        "Proposition de valeur": [
            "Simplification de la facturation et de la gestion financiere",
            f"Tarification accessible ({prix or round(median_prix,0)}EUR/mois)",
            "Conformite automatique e-facture 2026" if zone and "france" in zone.lower() else "Integrations API natives",
        ],
        "Segments clients": [cible or "PME / TPE", "Freelances et auto-entrepreneurs", "ETI en croissance"],
        "Sources de revenus": [
            f"Abonnement mensuel/annuel ({prix or round(median_prix,0)}EUR)",
            "Tarification a l'usage (pay-per-invoice)" if has_usage else "Plans premium avec fonctionnalites avancees",
            "Integrateur / Partenaires revendeurs",
        ],
        "Canaux": ["SaaS web + application mobile", "Partenariats comptables et ERP", "Marketing digital / freemium"],
        "Activites cles": ["Developpement produit", "Conformite reglementaire", "Support client et onboarding"],
        "Ressources cles": ["Equipe tech + data", "Dataset SaaS analyse", "Infrastructure cloud scalable"],
        "Partenaires cles": ["Experts-comptables", "Banques et etablissements de paiement", "PDP / Operateurs de facture electronique"],
        "Structure des couts": ["Developpement et maintenance", "Infrastructure cloud", "Marketing et acquisition clients"],
    }
    return canvas
