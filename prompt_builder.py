# prompt_builder.py
import pandas as pd
from dataset_loader import load_dataset, clean_dataset
from analytics import (analyser_segment, detecter_segment, scorer_viabilite,
                        detecter_ocean_bleu, recommandations_mvp, audit_conformite)

df = clean_dataset(load_dataset())

GREETINGS = {
    "hi","hello","hey","hiya","howdy",
    "bonjour","salut","bonsoir","bjr","slt",
    "marhaba","salam","ahlan","hola","ciao","hallo","ola"
}

STARTUP_KEYWORDS = [
    "lancer","startup","creer","business","mvp","fondateur","pitch","investisseur",
    "business model","positionnement","concurrents","marche","valider","launch",
    "entrepreneur","viabilite","pricing","strategie","compete","market","projet"
]

OCEAN_KEYWORDS = ["ocean bleu","niche","opportunite","gap","inexplore","blue ocean","differencier"]
SCORE_KEYWORDS = ["score","viabilite","viable","coherent","note ma","evaluer mon","analyser mon"]
FEATURE_KEYWORDS = ["feature importance","facteur","priorite mvp","quoi developper","roadmap","focus"]
CONFORMITE_KEYWORDS = ["conformite","reglementation","e-facture","2026","pdp","ppf","legal"]

def is_greeting(text):
    return text.strip().lower().rstrip("!,.?") in GREETINGS

def is_startup_query(text):
    q = text.lower()
    return any(kw in q for kw in STARTUP_KEYWORDS)

def build_prompt(question: str, lang: str = "fr") -> str:
    if is_greeting(question):
        return question

    lang_tag = "FRENCH" if lang == "fr" else "ENGLISH"
    header = f"[RESPOND IN {lang_tag} ONLY]\n\nUser question: {question}\n\n"
    q = question.lower()

    # ---- Detecteur d'oceans bleus ----
    if any(kw in q for kw in OCEAN_KEYWORDS):
        oceans = detecter_ocean_bleu(df)
        data_lines = ["=== Detecteur d'Oceans Bleus — Niches sous-exploitees ==="]
        if oceans:
            for o in oceans:
                opp_str = ", ".join(o["opportunites"]) if o["opportunites"] else "segment peu adresse"
                data_lines.append(
                    f"Segment {o['zone']} + {o['cible']} : {o['n_concurrents']} concurrents, "
                    f"satisfaction {o['satisfaction']}% — Opportunites: {opp_str}"
                )
        else:
            data_lines.append("Aucun ocean bleu clairement identifie sur ce dataset.")
        suffix = "\n\nPresent these as strategic opportunities. Explain WHY each represents an underserved niche and what product angle to take."
        return header + "\n".join(f"- {l}" for l in data_lines) + suffix

    # ---- Feature Importance / MVP ----
    if any(kw in q for kw in FEATURE_KEYWORDS):
        recs = recommandations_mvp()
        data_lines = ["=== Feature Importance — Facteurs de succes classes par impact predictif ==="]
        data_lines += recs
        suffix = "\n\nTranslate these into a concrete MVP roadmap. Explain what to build first and why based on statistical impact."
        return header + "\n".join(f"- {l}" for l in data_lines) + suffix

    # ---- Conformite reglementaire ----
    if any(kw in q for kw in CONFORMITE_KEYWORDS):
        zone, cible = detecter_segment(question)
        alertes = audit_conformite(df, zone=zone or "France", cible=cible)
        if not alertes:
            alertes = audit_conformite(df, zone="France", cible=None)
        data_lines = ["=== Audit de Conformite Reglementaire 2026 ==="] + alertes
        suffix = "\n\nExplain the regulatory context clearly and give actionable steps to achieve compliance."
        return header + "\n".join(f"- {l}" for l in data_lines) + suffix

    # ---- Score de viabilite ----
    if any(kw in q for kw in SCORE_KEYWORDS):
        import re
        zone, cible = detecter_segment(question)
        prix_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:eur|euro|€|usd|\$)?", q)
        prix = float(prix_match.group(1).replace(",", ".")) if prix_match else None
        if prix:
            has_freemium = any(w in q for w in ["freemium", "gratuit", "free"])
            has_usage = any(w in q for w in ["usage", "consommation", "usage-based"])
            alertes, score = scorer_viabilite(df, prix, zone, cible, has_freemium, has_usage)
            data_lines = [f"=== Simulateur de Positionnement — Score: {score}/10 ==="] + alertes
        else:
            data_lines = ["Pour calculer le score de viabilite, precisez votre prix envisage (ex: 29EUR/mois)."]
        suffix = "\n\nPresent the viability score clearly with specific actionable recommendations for each alert."
        return header + "\n".join(f"- {l}" for l in data_lines) + suffix

    # ---- Mode conseil strategique startup ----
    if is_startup_query(question):
        zone, cible = detecter_segment(question)
        data_lines = []
        if zone or cible:
            insights = analyser_segment(df, zone=zone, cible=cible)
            if insights:
                label = " + ".join(filter(None, [zone, cible])) or "general"
                data_lines.append(f"=== Analyse strategique du segment: {label} ===")
                data_lines += insights
                # Ajouter conformite si France
                if zone and "france" in zone.lower():
                    conformite = audit_conformite(df, zone=zone, cible=cible)
                    if conformite:
                        data_lines.append("--- Alerte Reglementaire ---")
                        data_lines += conformite
                # Ajouter top 3 feature importance
                fi_recs = recommandations_mvp()
                if fi_recs:
                    data_lines.append("--- Top 3 facteurs de succes MVP ---")
                    data_lines += fi_recs[:3]
            else:
                data_lines += [f"Segment {zone}/{cible} non trouve. Donnees globales:"]
                data_lines += [f"Dataset : {len(df)} logiciels SaaS",
                               f"Zones : {df['zone'].value_counts().to_dict()}"]
        else:
            data_lines += [
                f"Vue globale du marche SaaS ({len(df)} logiciels) :",
                f"Zones couvertes : {df['zone'].value_counts().to_dict()}",
                f"Prix min median : {df['prix_min_eur'].median():.2f} EUR",
                f"Freemium : {df['a_freemium'].mean()*100:.0f}% des acteurs",
                f"Usage-based : {df['a_usage_based'].mean()*100:.0f}% des acteurs",
                f"Satisfaction moyenne : {df['satisfaction_pct'].mean():.1f}%",
            ]
        suffix = "\n\nAct as a strategic SaaS advisor. Translate data into actionable insights. Flag risks, opportunities, and concrete next steps. End with one focused follow-up question."
        return header + "\n".join(f"- {l}" for l in data_lines) + suffix

    # ---- Questions standard sur le dataset ----
    data_lines = []
    if "freemium" in q or "plan gratuit" in q:
        sub = df[df["a_freemium"] == 1]
        data_lines += [f"Logiciels avec freemium : {len(sub)} sur {len(df)}",
                       f"Note G2 moyenne : {sub['note_g2'].mean():.2f}",
                       f"Prix min moyen : {sub['prix_min_eur'].mean():.2f} EUR"]
    elif "essai" in q or "trial" in q:
        sub = df[df["a_essai_gratuit"] == 1]
        data_lines += [f"Logiciels avec essai gratuit : {len(sub)} sur {len(df)}",
                       f"Note G2 moyenne : {sub['note_g2'].mean():.2f}"]
    elif "france" in q:
        sub = df[df["zone"] == "France"]
        data_lines += [f"Logiciels en France : {len(sub)} sur {len(df)}",
                       f"Prix min moyen : {sub['prix_min_eur'].mean():.2f} EUR",
                       f"Note G2 moyenne : {sub['note_g2'].mean():.2f}",
                       f"Conformes e-facture 2026 : {sub[sub['conforme_e_facture_2026']==1].shape[0]}"]
    elif "maroc" in q or "afrique" in q:
        sub = df[df["zone"] == "Mondial"]
        data_lines += [f"Zones disponibles : {df['zone'].value_counts().to_dict()}",
                       f"Logiciels zone Mondiale : {len(sub)}"]
    elif "facture" in q or "e-facture" in q:
        sub = df[df["conforme_e_facture_2026"] == 1]
        data_lines += [f"Logiciels conformes e-facture 2026 : {len(sub)} sur {len(df)}",
                       f"Avec PDP : {sub[sub['has_PDP']==1].shape[0]}",
                       f"Prix min moyen : {sub['prix_min_eur'].mean():.2f} EUR"]
    elif "api" in q:
        sub = df[df["has_API"] == 1]
        data_lines += [f"Logiciels avec API : {len(sub)} sur {len(df)}",
                       f"Note G2 moyenne : {sub['note_g2'].mean():.2f}"]
    elif "crm" in q:
        sub = df[df["has_CRM"] == 1]
        data_lines += [f"Logiciels avec CRM : {len(sub)} sur {len(df)}",
                       f"Prix min moyen : {sub['prix_min_eur'].mean():.2f} EUR"]
    elif "mobile" in q:
        sub = df[df["has_mobile"] == 1]
        data_lines += [f"Logiciels avec app mobile : {len(sub)} sur {len(df)}"]
    elif any(w in q for w in ["prix","tarif","cout","cher","budget","price","cost"]):
        data_lines += [f"Prix min : median {df['prix_min_eur'].median():.2f} EUR, moy. {df['prix_min_eur'].mean():.2f} EUR",
                       f"Classes de prix : {df['Classe_prix'].value_counts().to_dict()}"]
    elif any(w in q for w in ["note","avis","satisfaction","meilleur","top","best","rating"]):
        top5 = df.nlargest(5,"note_g2")[["nom_logiciel","note_g2"]].to_dict("records") if "nom_logiciel" in df.columns else []
        data_lines += [f"Note G2 moyenne : {df['note_g2'].mean():.2f}, satisfaction {df['satisfaction_pct'].mean():.1f}%"]
        if top5:
            data_lines.append("Top 5 : " + str(top5))
    elif any(w in q for w in ["financement","levee","investissement","funding"]):
        funded = df[df["total_financement_mUSD"] > 0]
        data_lines += [f"Logiciels avec financement : {len(funded)} sur {len(df)}",
                       f"Total : {funded['total_financement_mUSD'].sum():.1f}M USD"]
    else:
        data_lines += [f"Dataset SaaS : {len(df)} logiciels",
                       f"Zones : {df['zone'].value_counts().to_dict()}",
                       f"Note G2 moyenne : {df['note_g2'].mean():.2f}",
                       f"Prix min moyen : {df['prix_min_eur'].mean():.2f} EUR"]

    return (header + "Relevant data:\n"
            + "\n".join(f"- {l}" for l in data_lines)
            + "\n\nAnswer concisely and focus on the question only.")

# ============================================================
# Helpers pour les nouveaux modules (injectes dans build_prompt)
# ============================================================

SWOT_KEYWORDS = ["swot","forces","faiblesses","opportunites","menaces","analyse strategique","strengths"]
BENCHMARK_KEYWORDS = ["benchmark","comparer mon","comparaison","tableau comparatif","vs marche","par rapport au marche"]
REVENUE_KEYWORDS = ["mrr","arr","revenu","chiffre d affaire","revenue","clients par mois","simuler mes revenus","combien je gagne"]
CANVAS_KEYWORDS = ["business model canvas","bmc","canvas","proposition de valeur","modele economique","canaux distribution"]
SCORE_GLOBAL_KEYWORDS = ["score global","note globale","score final","evaluer globalement","multi-critere"]

# Patcher build_prompt pour intercepter ces nouveaux mots-cles AVANT le return existant
_original_build_prompt = build_prompt

def build_prompt(question: str, lang: str = "fr") -> str:
    import re
    from analytics import (benchmark_concurrents, score_global_viabilite,
                            analyse_swot, simulateur_revenus, business_model_canvas)

    if is_greeting(question):
        return question

    lang_tag = "FRENCH" if lang == "fr" else "ENGLISH"
    header = f"[RESPOND IN {lang_tag} ONLY]\n\nUser question: {question}\n\n"
    q = question.lower()

    # --- SWOT ---
    if any(kw in q for kw in SWOT_KEYWORDS):
        zone, cible = detecter_segment(question)
        prix_m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:eur|euro|€)?", q)
        prix = float(prix_m.group(1).replace(",", ".")) if prix_m else None
        swot = analyse_swot(df, zone=zone, cible=cible, prix=prix,
                             has_mobile="mobile" in q, has_api="api" in q,
                             has_freemium="freemium" in q, has_usage="usage" in q)
        data_lines = ["=== Analyse SWOT automatique ==="]
        for k, v in swot.items():
            data_lines.append(f"** {k} **")
            data_lines += [f"  + {item}" for item in v]
        suffix = "\n\nPresent this as a professional SWOT analysis. For each item explain the strategic implication."
        return header + "\n".join(f"- {l}" for l in data_lines) + suffix

    # --- Benchmark ---
    if any(kw in q for kw in BENCHMARK_KEYWORDS):
        zone, cible = detecter_segment(question)
        prix_m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:eur|euro|€)?", q)
        prix = float(prix_m.group(1).replace(",", ".")) if prix_m else None
        bm = benchmark_concurrents(df, zone=zone, cible=cible, prix=prix,
                                    has_mobile="mobile" in q, has_api="api" in q,
                                    has_freemium="freemium" in q)
        data_lines = [f"=== Benchmark vs Marche ({bm.get('n', 0)} concurrents) ===",
                      f"Score d'alignement : {bm.get('alignment', 0)}%"]
        for row in bm.get("rows", []):
            data_lines.append(f"{row[0]} | Projet: {row[1]} | Marche: {row[2]}")
        suffix = "\n\nPresent this as a competitive benchmark table. Explain the alignment score and give specific recommendations for each gap."
        return header + "\n".join(f"- {l}" for l in data_lines) + suffix

    # --- Score Global ---
    if any(kw in q for kw in SCORE_GLOBAL_KEYWORDS):
        zone, cible = detecter_segment(question)
        prix_m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:eur|euro|€)?", q)
        prix = float(prix_m.group(1).replace(",", ".")) if prix_m else None
        scores, final = score_global_viabilite(df, zone=zone, cible=cible, prix=prix,
                                                has_mobile="mobile" in q, has_api="api" in q,
                                                has_freemium="freemium" in q, has_usage="usage" in q,
                                                target_france=zone and "france" in zone.lower())
        data_lines = [f"=== Score Global de Viabilite : {final}/10 ==="]
        for dim, s in scores.items():
            data_lines.append(f"{dim} : {s}/10")
        suffix = "\n\nPresent each dimension score with a clear explanation. Give 1-2 concrete improvement actions per weak dimension. Conclude with the overall verdict."
        return header + "\n".join(f"- {l}" for l in data_lines) + suffix

    # --- MRR / ARR ---
    if any(kw in q for kw in REVENUE_KEYWORDS):
        prix_m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:eur|euro|€)?", q)
        clients_m = re.search(r"(\d+)\s*(?:clients?|abonnes?|users?)", q)
        prix = float(prix_m.group(1).replace(",", ".")) if prix_m else 15.0
        clients = int(clients_m.group(1)) if clients_m else 100
        rev = simulateur_revenus(prix, clients)
        data_lines = [
            f"=== Simulateur de Revenus ===",
            f"Prix mensuel : {prix}EUR | Clients : {clients}",
            f"MRR (Monthly Recurring Revenue) : {rev['mrr']:,.2f}EUR",
            f"ARR (Annual Recurring Revenue) : {rev['arr']:,.2f}EUR",
            f"LTV estimee (churn {rev['churn_rate']}%) : {rev['ltv']:,.2f}EUR",
            f"LTV/CAC ratio : {rev['ltv_cac']:.1f}x",
            f"Payback period : {rev['payback_mois']:.0f} mois",
        ]
        suffix = "\n\nExplain each metric clearly. Interpret the LTV/CAC ratio (>3 is good). Give concrete levers to improve these metrics."
        return header + "\n".join(f"- {l}" for l in data_lines) + suffix

    # --- Business Model Canvas ---
    if any(kw in q for kw in CANVAS_KEYWORDS):
        zone, cible = detecter_segment(question)
        prix_m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:eur|euro|€)?", q)
        prix = float(prix_m.group(1).replace(",", ".")) if prix_m else None
        canvas = business_model_canvas(df, zone=zone, cible=cible, prix=prix, has_usage="usage" in q)
        data_lines = ["=== Business Model Canvas automatique ==="]
        for bloc, items in canvas.items():
            data_lines.append(f"** {bloc} **")
            data_lines += [f"  - {item}" for item in items]
        suffix = "\n\nPresent this as a professional Business Model Canvas. For each bloc explain the strategic rationale based on market data."
        return header + "\n".join(f"- {l}" for l in data_lines) + suffix

    # Fallback to original logic
    return _original_build_prompt(question, lang)
