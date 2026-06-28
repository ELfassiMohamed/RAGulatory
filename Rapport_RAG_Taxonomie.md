# Rapport — Systeme RAG & Taxonomie Batiment

## 1. Objectif du RAG

Le RAG (Retrieval-Augmented Generation) de RAGulatory permet d'interroger en langage naturel une base de connaissances reglementaires et d'obtenir des reponses sourcees. Deux domaines :

- **Conformite SaaS** : RGPD, CNDP Maroc, SLA, securite
- **Reglementation Batiment Maroc** : classification via Taxonomie V1

## 2. Fonctionnement et integration Taxonomie

### Pipeline

```
Question → TF-IDF (scratch) → Cosine Similarity → Top-K → LLM → Reponse
```

### TF-IDF from scratch — `rag_conformite.py`

```python
def _tokenize(text: str) -> list:
    """Tokenisation : minuscules, mots de 3+ caracteres."""
    return re.findall(r'[a-záàâäéèêëîïôùûüçñ]{3,}', text.lower())


def _build_tfidf_index(docs: list) -> tuple:
    N = len(docs)
    df = {}
    tokenized_docs = []
    for doc in docs:
        tokens = _tokenize(doc["text"])
        tokenized_docs.append(tokens)
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    idf = {t: math.log(N / df[t]) for t in df}

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
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
```

### Chargement Taxonomie depuis Excel — `rag_conformite.py`

```python
def _load_taxonomy_base() -> list:
    path = "Taxonomie_V-1.xlsx"
    xls = pd.ExcelFile(path)
    entries = []

    # Sheet 1 : Taxonomie (55 lignes)
    df_taxo = pd.read_excel(xls, "Taxonomie")
    for i, row in df_taxo.iterrows():
        parts = []
        if pd.notna(row.get("Sous-domaine")):   parts.append(str(row["Sous-domaine"]).strip())
        if pd.notna(row.get("Catégorie")):       parts.append(str(row["Catégorie"]).strip())
        if pd.notna(row.get("Caracteristiques")): parts.append(str(row["Caracteristiques"]).strip())
        if not parts:
            continue
        entries.append({
            "id": f"taxo_{i}",
            "source": f"Taxonomie Batiment V1 — {row.get('Domaine', 'Non classe')}",
            "zone": "Reglementation Batiment",
            "text": " — ".join(parts)
        })

    # Sheet 2 : Glossaire (33 lignes)
    df_gloss = pd.read_excel(xls, "Glossaire")
    for i, row in df_gloss.iterrows():
        if pd.isna(row.get("Definition")):
            continue
        entries.append({
            "id": f"glossaire_{i}",
            "source": "Taxonomie Batiment V1 — Glossaire",
            "zone": "Reglementation Batiment",
            "text": f"{row['Acronyme / Terme']} : {row['Definition']}"
        })

    return entries  # 81 entrees au total
```

### Indexation automatique au demarrage

```python
class ConformiteRAG:
    def __init__(self, docs=None):
        self.docs = docs or KNOWLEDGE_BASE
        self.tfidf_index, self.idf = _build_tfidf_index(self.docs)
        self.client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    def retrieve(self, query: str, top_k: int = 3, zone_filter: str = None) -> list:
        query_vec = _vectorize_query(query, self.idf)
        scores = []
        for i, doc in enumerate(self.docs):
            if zone_filter and doc["zone"] != zone_filter and doc["zone"] != "International":
                continue
            score = _cosine_similarity(query_vec, self.tfidf_index[i])
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [{"doc": self.docs[i], "score": s} for s, i in scores[:top_k]]
```

---

## 3. Comment MCDM decide

### Criteres et poids — `taxonomie_advisor.py`

```python
BUILDING_CRITERIA = [
    "surface_utilisable",      # 20%  | Maximiser
    "cout_estime",             # 25%  | Minimiser
    "conformite_reglementaire",# 25%  | Maximiser
    "faisabilite_technique",   # 15%  | Maximiser
    "delai_realisation",       # 10%  | Minimiser
    "ml_viability_score"       #  5%  | Maximiser
]
```

### Normalisation et score WSM — `mcdm.py`

```python
def _normalize_minmax(value, c_min, c_max, criteria_type="max"):
    if c_max == c_min:
        return 1.0
    if criteria_type == "max":
        return (value - c_min) / (c_max - c_min)
    else:  # min : plus la valeur est basse, mieux c'est
        return (c_max - value) / (c_max - c_min)


def wsm(solutions, criteria, weights, criteria_types=None):
    bounds = _get_bounds(solutions, criteria)
    results = []

    for sol in solutions:
        total = 0.0
        for c in criteria:
            raw = sol["criteria_values"].get(c, 0)
            c_min, c_max = bounds[c]
            n = _normalize_minmax(raw, c_min, c_max, criteria_types.get(c, "max"))
            total += n * weights.get(c, 0)

        results.append({
            "name": sol["name"],
            "score": round(total, 3),
            "recommended": False
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    results[0]["recommended"] = True
    return results
```

### Methode TOPSIS — `mcdm.py`

```python
def topsis(solutions, criteria, weights, criteria_types=None):
    # Matrice normalisee vectoriellement
    norm_matrix = [[matrix[i][j] / col_norms[j] for j in range(n_crit)]
                   for i in range(n_sol)]

    # Matrice ponderee
    weighted = [[norm_matrix[i][j] * w_list[j] for j in range(n_crit)]
                for i in range(n_sol)]

    # Solutions ideales A+ et A-
    a_pos = [max(col) if ctype == "max" else min(col) for col, ctype in ...]
    a_neg = [min(col) if ctype == "max" else max(col) for col, ctype in ...]

    # Distance euclidienne + score de proximite Ci
    results = []
    for i in range(n_sol):
        d_pos = math.sqrt(sum((weighted[i][j] - a_pos[j]) ** 2 for j in range(n_crit)))
        d_neg = math.sqrt(sum((weighted[i][j] - a_neg[j]) ** 2 for j in range(n_crit)))
        ci = d_neg / (d_pos + d_neg) if (d_pos + d_neg) > 0 else 0.0
        results.append({"name": solutions[i]["name"], "score": round(ci, 3)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
```

### Pipeline complet — `taxonomie_advisor.py`

```python
def recommend(query: str, method: str = "wsm") -> dict:
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    # 1. Extraction des parametres terrain (LLM)
    params = _extract_parameters(query, client)

    # 2. RAG Taxonomie (TF-IDF + cosine similarity)
    regulations = _retrieve_regulations(params)

    # 3. Generation de 3-5 plans (LLM)
    plans = _generate_plans(params, regulations, client)

    # 4. ML Predictor + Conformite pour chaque plan
    for plan in plans:
        _enrich_plan(plan, params, client)

    # 5. Classement MCDM (WSM ou TOPSIS)
    ranking = _rank_plans(plans, method)

    return {
        "query": query,
        "terrain": params,
        "regulations": regulations,
        "plans": ranking["plans"],
        "mcdm": ranking["mcdm"],
    }
```

---

## 4. Stack technique et modeles IA

### Stack

| Couche | Technologie |
|--------|------------|
| Backend | Python 3.11 + Flask |
| Frontend | HTML/CSS/JS vanilla |
| LLM | OpenRouter (`openrouter/free`) |
| Retrieval | TF-IDF from scratch (math.log, re) |
| ML | 4× Random Forest (sklearn) |
| MCDM | WSM + TOPSIS (maison) |
| Donnees | pandas, Excel |

### Modeles

| Modele | Role | Precision |
|--------|------|:---------:|
| TF-IDF (scratch) | Retrieval textuel | Deterministe |
| Llama 3.1 8B / Qwen (OpenRouter) | Generation reponses, extraction, plans | Variable |
| Random Forest `classe_prix` | Prediction gamme de prix | 89.8% |
| Random Forest `efacture` | Prediction conformite e-facture | 87.0% |
| Random Forest `satisfaction` | Prediction satisfaction client | 77.5% |
| Random Forest `segment_taille` | Prediction segment taille | 70.5% |
| WSM / TOPSIS | Classement multicritere des plans | Deterministe |

### ML Predictor — entrainement Random Forest — `ml_predictor.py`

```python
class MLPredictor:
    def __init__(self):
        self.models = {}       # {task: RandomForestClassifier}
        self.encoders = {}     # {task: {col: LabelEncoder}}
        self.le_target = {}    # {task: LabelEncoder} pour la cible
        self.X_means = {}      # moyennes des features
        self.feature_names = {}
        self._is_trained = False

    def train(self):
        df = pd.read_excel("dataset_final.xlsx")
        df = df.drop_duplicates()
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("Inconnu")

        for task, meta in TASK_META.items():
            self._train_one(df, task, meta)

        self._is_trained = True

    def _train_one(self, df, task, meta):
        # − Preparation des features −
        cat_cols = [c for c in CAT_COLS if c in df.columns]
        if task != "segment_taille" and "segment_taille" in df.columns:
            cat_cols = cat_cols + ["segment_taille"]

        for col in cat_cols:
            if col in df.columns and col != meta["target"]:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[task] = {col: le}

        # − Encodage de la cible −
        le_t = LabelEncoder()
        df[meta["target"] + "_enc"] = le_t.fit_transform(df[meta["target"]].astype(str))
        self.le_target[task] = le_t

        # − Separation X/y −
        to_drop = [meta["target"], meta["target"] + "_enc"] + meta.get("drop_extra", [])
        X = df.drop(columns=to_drop, errors="ignore").select_dtypes(include=[np.number])
        y = df[meta["target"] + "_enc"]

        # − SMOTE (sur-echantillonnage) si disponible −
        if HAS_SMOTE and len(y.unique()) > 1:
            X, y = SMOTE(random_state=42).fit_resample(X, y)

        # − Entrainement Random Forest −
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        # − Evaluation −
        acc = (rf.predict(X_test) == y_test).mean()
        print(f"[ML] {task:<20} Accuracy={acc:.4f}  classes={list(le_t.classes_)}")
        self.models[task] = rf

    def predict_all(self, features: dict) -> dict:
        predictions = {}
        for task, model in self.models.items():
            pred = model.predict([list(features.values())])[0]
            proba = max(model.predict_proba([list(features.values())])[0])
            label = self.le_target[task].inverse_transform([pred])[0]
            predictions[task] = {"label": label, "confidence": round(proba * 100, 1)}
        return predictions
```

---

*Document genere le 16 juin 2026 — RAGulatory*
