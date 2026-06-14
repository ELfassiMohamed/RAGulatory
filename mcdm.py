"""
mcdm.py — Moteur de comparaison multicritère (MCDM)
Implémente WSM (Weighted Sum Model) et TOPSIS pour la comparaison de solutions SaaS.

Usage:
    from mcdm import wsm, topsis, DEFAULT_SAAS_SOLUTIONS, DEFAULT_CRITERIA
"""

import math

# ============================================================
# Données de référence (solutions SaaS par défaut)
# ============================================================

DEFAULT_CRITERIA = ["fonctionnalite", "cout_tco", "securite_sla"]

DEFAULT_CRITERIA_LABELS = {
    "fonctionnalite": "Adéquation Fonctionnelle",
    "cout_tco":       "Coût / TCO",
    "securite_sla":   "Sécurité & SLA"
}

DEFAULT_CRITERIA_TYPES = {
    "fonctionnalite": "max",   # plus c'est élevé, mieux c'est
    "cout_tco":       "min",   # plus c'est bas, mieux c'est
    "securite_sla":   "max"
}

DEFAULT_SAAS_SOLUTIONS = [
    {
        "name": "SaaS Alpha",
        "description": "Solution complète, bien intégrée, prix premium",
        "criteria_values": {
            "fonctionnalite": 90,
            "cout_tco":       850,  # EUR/mois (total)
            "securite_sla":   99.9
        }
    },
    {
        "name": "SaaS Beta",
        "description": "Très riche fonctionnellement, prix moyen",
        "criteria_values": {
            "fonctionnalite": 95,
            "cout_tco":       650,
            "securite_sla":   99.5
        }
    },
    {
        "name": "SaaS Gamma",
        "description": "Économique, fonctionnalités de base",
        "criteria_values": {
            "fonctionnalite": 70,
            "cout_tco":       290,
            "securite_sla":   99.0
        }
    }
]

# ============================================================
# Utilitaires communs
# ============================================================

def _get_bounds(solutions, criteria):
    """Calcule min/max de chaque critère sur l'ensemble des solutions."""
    bounds = {}
    for c in criteria:
        vals = [s["criteria_values"].get(c, 0) for s in solutions]
        bounds[c] = (min(vals), max(vals))
    return bounds


def _normalize_minmax(value, c_min, c_max, criteria_type="max"):
    """Normalisation min-max sur [0, 1]."""
    if c_max == c_min:
        return 1.0
    if criteria_type == "max":
        return (value - c_min) / (c_max - c_min)
    else:  # min: plus la valeur est basse, plus le score est haut
        return (c_max - value) / (c_max - c_min)


# ============================================================
# Méthode 1 : WSM — Weighted Sum Model
# ============================================================

def wsm(solutions, criteria, weights, criteria_types=None):
    """
    Weighted Sum Model — recommandation rapide et transparente.

    Étapes :
      1. Normalisation min-max des critères
      2. Application des poids
      3. Somme pondérée → score global

    Args:
        solutions      : liste de dicts {"name", "criteria_values": {crit: float}}
        criteria       : liste ordonnée des noms de critères
        weights        : dict {critère: poids} — doit sommer à 1
        criteria_types : dict {critère: "max"|"min"} (défaut : tous "max")

    Returns:
        list[dict] trié par score décroissant :
            {"name", "normalized": {crit: score}, "weighted": {crit: score}, "score", "recommended"}
    """
    if not solutions or not criteria:
        return []

    if criteria_types is None:
        criteria_types = {c: "max" for c in criteria}

    bounds = _get_bounds(solutions, criteria)
    results = []

    for sol in solutions:
        norm = {}
        weighted_scores = {}
        total = 0.0

        for c in criteria:
            raw = sol["criteria_values"].get(c, 0)
            c_min, c_max = bounds[c]
            n = _normalize_minmax(raw, c_min, c_max, criteria_types.get(c, "max"))
            w = weights.get(c, 0)
            norm[c] = round(n, 3)
            weighted_scores[c] = round(n * w, 3)
            total += n * w

        results.append({
            "name":        sol["name"],
            "description": sol.get("description", ""),
            "raw":         {c: sol["criteria_values"].get(c, 0) for c in criteria},
            "normalized":  norm,
            "weighted":    weighted_scores,
            "score":       round(total, 3),
            "recommended": False
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    if results:
        results[0]["recommended"] = True
    return results


# ============================================================
# Méthode 2 : TOPSIS
# ============================================================

def topsis(solutions, criteria, weights, criteria_types=None):
    """
    TOPSIS — Technique for Order of Preference by Similarity to Ideal Solution.

    Étapes :
      1. Normalisation vectorielle
      2. Matrice pondérée-normalisée
      3. Solutions idéales positive (A+) et négative (A-)
      4. Distances euclidiennes à A+ et A-
      5. Proximité relative Ci = Si- / (Si+ + Si-)

    Args: (identiques à wsm)

    Returns:
        list[dict] trié par Ci décroissant :
            {"name", "score" (Ci), "d_pos", "d_neg", "recommended"}
    """
    if not solutions or not criteria:
        return []

    if criteria_types is None:
        criteria_types = {c: "max" for c in criteria}

    n_sol = len(solutions)
    n_crit = len(criteria)

    # Matrice brute
    matrix = [[sol["criteria_values"].get(c, 0) for c in criteria] for sol in solutions]

    # --- Étape 1 : normalisation vectorielle ---
    col_norms = []
    for j in range(n_crit):
        sq_sum = sum(matrix[i][j] ** 2 for i in range(n_sol))
        col_norms.append(math.sqrt(sq_sum) if sq_sum > 0 else 1.0)

    norm_matrix = [
        [matrix[i][j] / col_norms[j] for j in range(n_crit)]
        for i in range(n_sol)
    ]

    # --- Étape 2 : matrice pondérée ---
    w_list = [weights.get(c, 0) for c in criteria]
    weighted = [
        [norm_matrix[i][j] * w_list[j] for j in range(n_crit)]
        for i in range(n_sol)
    ]

    # --- Étape 3 : solutions idéales ---
    a_pos, a_neg = [], []
    for j in range(n_crit):
        col = [weighted[i][j] for i in range(n_sol)]
        ctype = criteria_types.get(criteria[j], "max")
        if ctype == "max":
            a_pos.append(max(col))
            a_neg.append(min(col))
        else:
            a_pos.append(min(col))
            a_neg.append(max(col))

    # --- Étape 4 & 5 : distances et proximité ---
    results = []
    for i in range(n_sol):
        d_pos = math.sqrt(sum((weighted[i][j] - a_pos[j]) ** 2 for j in range(n_crit)))
        d_neg = math.sqrt(sum((weighted[i][j] - a_neg[j]) ** 2 for j in range(n_crit)))
        ci = d_neg / (d_pos + d_neg) if (d_pos + d_neg) > 0 else 0.0
        results.append({
            "name":        solutions[i]["name"],
            "description": solutions[i].get("description", ""),
            "raw":         {c: solutions[i]["criteria_values"].get(c, 0) for c in criteria},
            "score":       round(ci, 3),
            "d_pos":       round(d_pos, 4),
            "d_neg":       round(d_neg, 4),
            "recommended": False
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    if results:
        results[0]["recommended"] = True
    return results


# ============================================================
# Wrapper : compare() — choix automatique de méthode
# ============================================================

def compare(solutions=None, weights=None, method="wsm", criteria=None, criteria_types=None):
    """
    Point d'entrée principal.

    Args:
        solutions      : liste de solutions (utilise DEFAULT_SAAS_SOLUTIONS si None)
        weights        : poids par critère (utilise poids égaux si None)
        method         : "wsm" | "topsis"
        criteria       : liste de critères (utilise DEFAULT_CRITERIA si None)
        criteria_types : types (utilise DEFAULT_CRITERIA_TYPES si None)

    Returns:
        dict {"results": [...], "method": str, "criteria_labels": dict}
    """
    if solutions is None:
        solutions = DEFAULT_SAAS_SOLUTIONS

    if criteria is None:
        criteria = DEFAULT_CRITERIA

    if criteria_types is None:
        criteria_types = DEFAULT_CRITERIA_TYPES

    # Poids par défaut : égaux
    if weights is None:
        n = len(criteria)
        weights = {c: round(1 / n, 3) for c in criteria}
    else:
        # Normaliser pour que la somme = 1
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

    if method == "topsis":
        results = topsis(solutions, criteria, weights, criteria_types)
    else:
        results = wsm(solutions, criteria, weights, criteria_types)

    return {
        "results":         results,
        "method":          method,
        "criteria_labels": {c: DEFAULT_CRITERIA_LABELS.get(c, c) for c in criteria},
        "weights_used":    {c: round(weights.get(c, 0), 3) for c in criteria}
    }


# ============================================================
# Test rapide
# ============================================================

if __name__ == "__main__":
    weights = {"fonctionnalite": 0.30, "cout_tco": 0.40, "securite_sla": 0.30}

    print("=== WSM ===")
    r_wsm = compare(weights=weights, method="wsm")
    for sol in r_wsm["results"]:
        tag = " ← Recommandé" if sol.get("recommended") else ""
        print(f"  {sol['name']}: {sol['score']:.3f}{tag}")

    print("\n=== TOPSIS ===")
    r_topsis = compare(weights=weights, method="topsis")
    for sol in r_topsis["results"]:
        tag = " ← Recommandé" if sol.get("recommended") else ""
        print(f"  {sol['name']}: {sol['score']:.3f}{tag}")
