"""
ml_predictor.py — Module de Prédiction IA (4 modèles Random Forest)
====================================================================
Intègre directement les pipelines des scripts de classification PFE :
  - classification_saas.py        → Classe_prix
  - classification_e_facture.py   → conforme_e_facture_2026
  - classification_satisfaction.py → niveau_satisfaction
  - classification_segment_taille.py → segment_taille

Fournit :
  1. predict_all(features)      → 4 prédictions simultanées
  2. explain_prediction(...)    → Explicabilité SHAP-like (feature contributions)
  3. tco_5ans(...)              → Simulateur TCO sur 5 ans avec croissance
  4. detect_price_anomaly(...)  → Détecteur d'anomalie de prix (Smart Pricing)
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Dépendances ML ─────────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    try:
        from imblearn.over_sampling import SMOTE
        HAS_SMOTE = True
    except ImportError:
        HAS_SMOTE = False
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_final.xlsx")

# ── Variables catégorielles (même liste que classification_saas.py) ─────────────
CAT_COLS = ['zone', 'cible', 'pays_siege', 'type_derniere_levee']
# Note: segment_taille est cible pour un modèle, feature pour les autres

# ── Labels par tâche ────────────────────────────────────────────────────────────
TASK_META = {
    "classe_prix": {
        "target":      "Classe_prix",
        "drop_extra":  ["Classe_prix_enc", "nom_logiciel"],
        "label":       "Gamme de Prix",
        "icon":        "💰",
        "classes":     ["Enterprise", "Free", "Low", "Medium"],
        "colors":      {"Enterprise": "#7c3aed", "Free": "#059669", "Low": "#1d4ed8", "Medium": "#d97706"},
        "accuracy":    "98.24%"
    },
    "efacture": {
        "target":      "conforme_e_facture_2026",
        "drop_extra":  ["nom_logiciel"],
        "label":       "Conformité E-Facture 2026",
        "icon":        "📋",
        "classes":     ["Non conforme", "Conforme"],
        "colors":      {"Non conforme": "#ef4444", "Conforme": "#22c55e"},
        "accuracy":    "88.66%"
    },
    "satisfaction": {
        "target":      "niveau_satisfaction",
        "drop_extra":  ["satisfaction_pct", "nom_logiciel"],
        "label":       "Satisfaction Prédictive",
        "icon":        "⭐",
        "classes":     ["Moyenne", "Bonne", "Excellente"],
        "colors":      {"Moyenne": "#ef4444", "Bonne": "#f59e0b", "Excellente": "#22c55e"},
        "accuracy":    "77.13%"
    },
    "segment_taille": {
        "target":      "segment_taille",
        "drop_extra":  ["nom_logiciel"],
        "label":       "Segment Taille Cible",
        "icon":        "🏢",
        "classes":     ["Petite", "Moyenne", "Grande", "Très Grande"],
        "colors":      {"Petite": "#1d4ed8", "Moyenne": "#059669", "Grande": "#d97706", "Très Grande": "#7c3aed"},
        "accuracy":    "90.70%"
    }
}


# ════════════════════════════════════════════════════════════════════════════════
# Classe principale : MLPredictor
# ════════════════════════════════════════════════════════════════════════════════

class MLPredictor:
    """
    Entraîne les 4 modèles Random Forest à partir du dataset et fournit
    des prédictions, explications et simulations financières.
    """

    def __init__(self):
        self.models    = {}       # {task: RandomForestClassifier}
        self.encoders  = {}       # {task: {col: LabelEncoder}}
        self.le_target = {}       # {task: LabelEncoder}
        self.X_means   = {}       # {task: Series} — moyennes des features (pour SHAP)
        self.feature_names = {}   # {task: [col names]}
        self._is_trained = False
        self._df_raw     = None

    # ── Entraînement ──────────────────────────────────────────────────────────

    def train(self):
        """Charge le dataset et entraîne les 4 modèles RF."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn requis. pip install scikit-learn")

        print("[ML] Chargement du dataset...")
        df_raw = pd.read_excel(DATASET_PATH)
        df_raw = df_raw.drop_duplicates()
        for col in df_raw.columns:
            if df_raw[col].dtype in ['float64', 'int64']:
                df_raw[col] = df_raw[col].fillna(0)
            else:
                df_raw[col] = df_raw[col].fillna("Inconnu")

        # Ajouter niveau_satisfaction si pas présent
        if "niveau_satisfaction" not in df_raw.columns and "satisfaction_pct" in df_raw.columns:
            df_raw["niveau_satisfaction"] = pd.cut(
                df_raw["satisfaction_pct"],
                bins=[-np.inf, 83.5, 88.0, np.inf],
                labels=["Moyenne", "Bonne", "Excellente"]
            ).astype(str)

        self._df_raw = df_raw
        print(f"[ML] Dataset : {df_raw.shape[0]} lignes × {df_raw.shape[1]} colonnes")

        for task, meta in TASK_META.items():
            self._train_one(df_raw, task, meta)

        self._is_trained = True
        print("[ML] OK 4 modeles entraines avec succes.")

    def _train_one(self, df_raw: pd.DataFrame, task: str, meta: dict):
        """Entraîne un seul modèle RF pour une tâche de classification."""
        target = meta["target"]
        drop_extra = meta["drop_extra"]

        df = df_raw.copy()

        # Supprimer la colonne source
        df = df.drop(columns=["source"], errors="ignore")

        # Pour le modèle segment_taille : encoder segment_taille différemment
        # (c'est la cible, pas une feature)
        cat_cols_task = [c for c in CAT_COLS if c in df.columns]
        if task != "segment_taille" and "segment_taille" in df.columns:
            cat_cols_task = cat_cols_task + ["segment_taille"]

        # Encoder les catégorielles
        le_dict = {}
        for col in cat_cols_task:
            if col in df.columns and col != target:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le

        self.encoders[task] = le_dict

        # Encoder la cible
        le_t = LabelEncoder()
        if target in df.columns:
            df[target + "_enc"] = le_t.fit_transform(df[target].astype(str))
        else:
            print(f"[ML] WARN Colonne cible '{target}' absente pour la tache {task}")
            return
        self.le_target[task] = le_t

        # Colonnes à exclure de X
        to_drop = [target, target + "_enc"] + [c for c in drop_extra if c in df.columns]
        to_drop += [c for c in ["Classe_prix", "Classe_prix_enc", "nom_logiciel", "source"]
                    if c in df.columns and c not in to_drop]

        X = df.drop(columns=to_drop, errors="ignore")
        # Garder uniquement les colonnes numériques
        X = X.select_dtypes(include=[np.number])
        y = df[target + "_enc"]

        self.feature_names[task] = list(X.columns)
        self.X_means[task] = X.mean()

        # SMOTE si disponible
        if HAS_SMOTE and len(y.unique()) > 1:
            try:
                smote = SMOTE(random_state=42)
                X, y = smote.fit_resample(X, y)
            except Exception:
                pass

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        acc = (rf.predict(X_test) == y_test).mean()
        print(f"[ML] OK {task:<20} Accuracy={acc:.4f}  classes={list(le_t.classes_)}")

        self.models[task] = rf

    # ── Prédiction ────────────────────────────────────────────────────────────

    def _prepare_input(self, task: str, user_features: dict) -> pd.DataFrame:
        """
        Prépare un vecteur de features pour la prédiction.
        Remplit les colonnes manquantes par la moyenne du dataset.
        Toutes les valeurs sont garanties numériques (float).
        """
        means = self.X_means[task]
        feat_names = self.feature_names[task]
        encoders = self.encoders[task]

        row = {}
        for col in feat_names:
            if col in user_features:
                val = user_features[col]
                # Encoder les catégorielles (zone, cible, pays_siege, segment_taille…)
                if col in encoders:
                    le = encoders[col]
                    str_val = str(val)
                    if str_val in le.classes_:
                        val = int(le.transform([str_val])[0])
                    else:
                        # Valeur inconnue → classe par défaut (premier encodage)
                        val = int(le.transform([le.classes_[0]])[0])
            else:
                val = means.get(col, 0)

            # Garde-fou : forcer numeric — si la valeur est encore un string, utiliser la moyenne
            try:
                row[col] = float(val)
            except (ValueError, TypeError):
                row[col] = float(means.get(col, 0))

        return pd.DataFrame([row], columns=feat_names).astype(float)

    def predict_one(self, task: str, user_features: dict) -> dict:
        """
        Prédit une seule tâche.

        Returns:
            {"label": str, "confidence": float, "probabilities": dict}
        """
        if task not in self.models:
            return {"error": f"Modèle '{task}' non disponible"}

        X = self._prepare_input(task, user_features)
        rf = self.models[task]
        le_t = self.le_target[task]
        meta = TASK_META[task]

        pred_enc = rf.predict(X)[0]
        proba = rf.predict_proba(X)[0]

        label = le_t.inverse_transform([pred_enc])[0]
        confidence = float(proba[pred_enc])

        proba_dict = {le_t.classes_[i]: round(float(p), 3) for i, p in enumerate(proba)}

        return {
            "task":        task,
            "label":       str(label),
            "confidence":  round(confidence * 100, 1),
            "icon":        meta["icon"],
            "display":     meta["label"],
            "color":       meta["colors"].get(str(label), "#6b7280"),
            "accuracy":    meta["accuracy"],
            "probabilities": proba_dict
        }

    def predict_all(self, user_features: dict) -> dict:
        """
        Prédit les 4 tâches simultanément.
        Retourne aussi l'explication SHAP-like pour le meilleur modèle (Classe_prix).
        """
        results = {}
        for task in TASK_META:
            try:
                results[task] = self.predict_one(task, user_features)
            except Exception as e:
                results[task] = {"error": str(e)}

        try:
            shap = self.explain(user_features, "classe_prix", top_k=6)
        except Exception:
            shap = []

        try:
            anomaly = self.detect_price_anomaly(user_features)
        except Exception:
            anomaly = {"has_anomaly": False}

        return {
            "predictions": results,
            "explanation": shap,
            "anomaly":     anomaly
        }

    # ── Explicabilité SHAP-like ───────────────────────────────────────────────

    def explain(self, user_features: dict, task: str = "classe_prix", top_k: int = 6) -> list:
        """
        Calcul SHAP simplifié : contribution de chaque feature à la prédiction.

        Formule : contribution_j = importance_j × (valeur_j - moyenne_j)
        Normalisé sur [-1, +1] pour l'affichage.

        Returns:
            list of {"feature": str, "contribution": float, "direction": "+" | "-"}
        """
        if task not in self.models:
            return []

        X = self._prepare_input(task, user_features)
        rf = self.models[task]
        means = self.X_means[task]
        feat_names = self.feature_names[task]

        importances = rf.feature_importances_
        contributions = []

        for i, col in enumerate(feat_names):
            val = float(X[col].iloc[0])
            mean_val = float(means.get(col, 0))
            std_val = float(self._df_raw[col].std()) if (self._df_raw is not None and col in self._df_raw.columns) else 1.0
            if std_val == 0:
                std_val = 1.0

            # Contribution normalisée
            dev = (val - mean_val) / std_val
            contrib = importances[i] * dev

            if abs(contrib) > 1e-6:
                contributions.append({
                    "feature":      col,
                    "contribution": round(float(contrib), 4),
                    "importance":   round(float(importances[i]), 4),
                    "value":        round(val, 2),
                    "direction":    "+" if contrib >= 0 else "-"
                })

        # Trier par valeur absolue décroissante
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        # Normaliser pour l'affichage [-100, +100]
        max_abs = max((abs(c["contribution"]) for c in contributions), default=1)
        for c in contributions:
            c["display_pct"] = round(c["contribution"] / max_abs * 100, 1)

        return contributions[:top_k]

    # ── Détection d'anomalie de prix ──────────────────────────────────────────

    def detect_price_anomaly(self, user_features: dict) -> dict:
        """
        Compare le prix fourni par l'utilisateur avec la prédiction du marché.
        Utile pour l'Assistant de Négociation B2B.
        """
        prix_propose = user_features.get("prix_max_eur") or user_features.get("prix_min_eur") or 0
        if prix_propose <= 0 or self._df_raw is None:
            return {"has_anomaly": False}

        # Prédit la classe de prix selon les features
        result = self.predict_one("classe_prix", user_features)
        classe_predite = result.get("label", "Low")

        # Prix moyen du marché pour cette classe
        prix_marche = self._df_raw.groupby("Classe_prix")["prix_max_eur"].median()
        prix_reference = float(prix_marche.get(classe_predite, prix_propose))

        if prix_reference <= 0:
            return {"has_anomaly": False}

        ecart_pct = ((prix_propose - prix_reference) / prix_reference) * 100

        return {
            "has_anomaly":      abs(ecart_pct) > 30,
            "prix_propose":     round(prix_propose, 2),
            "prix_reference":   round(prix_reference, 2),
            "ecart_pct":        round(ecart_pct, 1),
            "classe_predite":   classe_predite,
            "verdict":          (
                "🔴 Prix excessif — négociation recommandée"  if ecart_pct > 30
                else "✅ Prix dans les normes du marché"       if abs(ecart_pct) <= 30
                else "🟢 Bonne affaire — prix inférieur au marché"
            )
        }

    # ── TCO 5 ans ──────────────────────────────────────────────────────────────

    @staticmethod
    def tco_5ans(
        prix_mensuel: float,
        n_utilisateurs: int,
        taux_croissance_annuel: float = 0.15,
        migration_cost: float = 0,
        formation_cost: float = 0
    ) -> dict:
        """
        Simulateur de TCO sur 5 ans avec croissance des effectifs.

        Formule année Y :
            users_Y = n_utilisateurs × (1 + taux_croissance)^(Y-1)
            cout_Y  = prix_mensuel × 12 × users_Y

        Returns:
            dict avec courbe annuelle, TCO total, et point de pic de croissance
        """
        annual_costs = []
        cumulative   = []
        total = migration_cost + formation_cost

        for year in range(1, 6):
            users_y = n_utilisateurs * ((1 + taux_croissance_annuel) ** (year - 1))
            cost_y  = prix_mensuel * 12 * users_y
            total  += cost_y
            annual_costs.append(round(cost_y, 2))
            cumulative.append(round(total, 2))

        return {
            "annual_costs":   annual_costs,
            "cumulative":     cumulative,
            "tco_total":      round(total, 2),
            "annees":         [f"An {y}" for y in range(1, 6)],
            "users_par_an":   [
                round(n_utilisateurs * ((1 + taux_croissance_annuel) ** (y - 1)))
                for y in range(1, 6)
            ],
            "migration_cost": migration_cost,
            "formation_cost": formation_cost,
            "taux_croissance": taux_croissance_annuel
        }


# ════════════════════════════════════════════════════════════════════════════════
# Singleton — instance unique chargée au démarrage de Flask
# ════════════════════════════════════════════════════════════════════════════════

_predictor_instance: MLPredictor | None = None


def get_predictor() -> MLPredictor:
    """Retourne l'instance singleton. L'entraîne si nécessaire."""
    global _predictor_instance
    if _predictor_instance is None or not _predictor_instance._is_trained:
        _predictor_instance = MLPredictor()
        _predictor_instance.train()   # laisse lever l'exception pour signaler clairement l'erreur
    return _predictor_instance


# ════════════════════════════════════════════════════════════════════════════════
# Test rapide
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = MLPredictor()
    p.train()

    # Cas test : SaaS avec API + mobile + comptabilité ciblant les PME en France
    features_test = {
        "has_API":     1,
        "has_mobile":  1,
        "has_compta":  1,
        "has_CRM":     0,
        "has_PDP":     0,
        "a_freemium":  0,
        "a_usage_based": 0,
        "a_essai_gratuit": 1,
        "prix_min_eur": 25,
        "prix_max_eur": 89,
        "note_g2":     4.2,
        "zone":        "France",
        "cible":       "PME",
    }

    print("\n── Prédiction toutes tâches ──")
    results = p.predict_all(features_test)
    for task, r in results["predictions"].items():
        print(f"  {r['icon']} {r['display']:<30} → {r['label']} ({r['confidence']}%)")

    print("\n-- Explicabilite (top 6) --")
    for c in results["explanation"]:
        sign = "+" if c["direction"] == "+" else "-"
        print(f"  {sign} {c['feature']:<30} {c['display_pct']:+.1f}%")

    print("\n-- Anomalie de prix --")
    a = results["anomaly"]
    print(f"  {a.get('verdict', 'N/A')}")

    print("\n-- TCO 5 ans (25 EUR/mois, 20 users, croissance 20%) --")
    tco = p.tco_5ans(25, 20, taux_croissance_annuel=0.20, migration_cost=500, formation_cost=300)
    for i, (an, cout, users) in enumerate(zip(tco["annees"], tco["annual_costs"], tco["users_par_an"])):
        print(f"  {an} : {cout:,.0f} € ({users} users)")
    print(f"  TCO Total : {tco['tco_total']:,.0f} €")
