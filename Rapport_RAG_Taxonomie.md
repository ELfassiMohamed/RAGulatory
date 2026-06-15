# Rapport — Système RAG & Taxonomie Ontologique Bâtiment

## 1. Structure du rapport

1. **Objectif** — Pourquoi ce RAG a été conçu
2. **Architecture** — Pipeline complet (Indexation → Retrieval → Génération)
3. **Modèles utilisés** — TF-IDF from scratch + LLM (OpenRouter)
4. **Base de connaissances** — Sources réglementaires et ontologiques
5. **Intégration de la Taxonomie Bâtiment** — Chargement et indexation
6. **Fonctionnalités dérivées** — 3 modes d'exploitation
7. **Limites et perspectives**

---

## 2. Objectif du système RAG

Le système RAG (Retrieval-Augmented Generation) de RAGulatory a pour but de fournir **des réponses sourcées et vérifiables** dans deux domaines réglementaires :

- **Conformité SaaS** : RGPD (Union Européenne), CNDP (Maroc), bonnes pratiques SLA et sécurité — pour accompagner les acheteurs et éditeurs de logiciels SaaS dans leurs obligations légales.
- **Réglementation Bâtiment** : classification des bâtiments selon la réglementation française (Habitation, ERP, ERT, EIC) à partir d'une taxonomie ontologique structurée. Cette taxonomie sert de référentiel unique pour déterminer les seuils réglementaires (nombre d'étages, effectifs, hauteur PBDN, etc.) applicables à un bâtiment.

Le RAG transforme une base documentaire hétérogène (articles de loi, glossaire technique, fiches de classification) en **un moteur de question-réponse intelligent** : l'utilisateur pose une question en langage naturel, le système retrouve les passages les plus pertinents, puis un LLM génère une réponse structurée et sourcée.

Contrairement à un chatbot classique qui repose uniquement sur la mémoire paramétrique du LLM, le RAG garantit que la réponse est **ancrée dans les documents de référence**, réduisant ainsi les risques d'hallucination et permettant de citer exactement la source réglementaire mobilisée.

---

## 3. Architecture — Pipeline complet

Le système repose sur un pipeline en 3 étapes, sans aucune dépendance externe lourde (pas de ChromaDB, Pinecone, ElasticSearch, ni sklearn) :

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1 : INDEXATION                                       │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────────┐  │
│  │ Documents │───→│ Tokenisation│───→│ Vecteurs TF-IDF     │  │
│  │ (90 docs) │    │ (mots ≥3   │    │ (dict {terme: poids})│  │
│  └──────────┘    │ caractères)│    └──────────────────────┘  │
│                   └───────────┘    + index IDF global        │
├─────────────────────────────────────────────────────────────┤
│  PHASE 2 : REQUÊTE                                           │
│  ┌──────────┐    ┌───────────┐    ┌──────────────┐          │
│  │ Question  │───→│ Vectorisation│───→│ Cosine         │          │
│  │ utilisateur│   │ (même IDF) │    │ Similarity    │          │
│  └──────────┘    └───────────┘    │ vs tous les   │          │
│                                    │ documents     │          │
│                                    └──────┬───────┘          │
│                                           ↓                  │
│                                    ┌──────────────┐          │
│                                    │ Top-K chunks │          │
│                                    │ (score > 0)  │          │
│                                    └──────────────┘          │
├─────────────────────────────────────────────────────────────┤
│  PHASE 3 : GÉNÉRATION                                        │
│  ┌──────────────────┐    ┌──────────────┐                   │
│  │ Contexte =        │───→│ LLM           │                   │
│  │ chunks concaténés │    │ (OpenRouter)  │                   │
│  │ + question        │    │ temperature=0.3│                   │
│  └──────────────────┘    └──────┬───────┘                   │
│                                  ↓                           │
│                          ┌──────────────────┐               │
│                          │ Réponse sourcée   │               │
│                          │ + sources citées  │               │
│                          └──────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### Détail des étapes

**3.1 Tokenisation** (`_tokenize`)
- Passage en minuscules
- Extraction des mots de 3 caractères ou plus (lettres latines accentuées incluses : áàâäéèêëîïôùûüçñ)
- Filtre les mots vides courts et la ponctuation

**3.2 TF-IDF from scratch** (`_build_tfidf_index`)

Implémentation native sans bibliothèque ML :

- **TF** (Term Frequency) : `tf(t, d) = occurrences du terme t dans le document d / nombre total de termes dans d`
- **IDF** (Inverse Document Frequency) : `idf(t) = log(N / df(t))` où `N` = nombre total de documents et `df(t)` = nombre de documents contenant t
- **Poids TF-IDF** : `w(t, d) = tf(t, d) × idf(t)`

Chaque document est représenté par un dictionnaire `{terme: poids_tfidf}`. L'index global est une liste de ces dictionnaires.

**3.3 Similarité cosinus** (`_cosine_similarity`)
```
cos(q, d) = somme(q_i × d_i) / (sqrt(somme(q_i²)) × sqrt(somme(d_i²)))
```
Le score est compris entre 0 (aucun terme commun) et 1 (documents identiques).

**3.4 Filtrage par zone**
Le système supporte le filtrage par zone réglementaire :
- `"Maroc"` — CNDP et réglementations marocaines
- `"UE"` — RGPD et réglementations européennes
- `"Réglementation Bâtiment"` — Taxonomie des bâtiments
- `"International"` — Bonnes pratiques générales (SLA, sécurité)

Les documents marqués `"International"` sont toujours inclus quel que soit le filtre de zone actif.

**3.5 Génération LLM**
Les top-K chunks (K=3 par défaut) sont concaténés dans un contexte structuré avec leurs sources, puis envoyés au LLM avec un prompt système spécialisé. Le prompt demande une réponse en 4 parties :
1. Verdict de risque (code couleur : ✅ / ⚠️ / 🔴)
2. Points réglementaires clés
3. Questions à poser à l'éditeur / professionnel
4. Recommandations concrètes

---

## 4. Modèles utilisés

### 4.1 TF-IDF — Implémentation native Python

- **Type** : Bag-of-words pondéré (statistique)
- **Bibliothèque** : Aucune — `math.log()` et `re.findall()` uniquement
- **Vocabulaire** : Construit dynamiquement à partir des 90 documents
- **Dimensionnalité** : Variable (égale au nombre de termes uniques dans le corpus, ≈200-500 termes pour 90 documents réglementaires)
- **Avantage** : Déterministe, reproductible, explicable, zéro dépendance
- **Inconvénient** : Pas de compréhension sémantique (ne capture pas les synonymes ni les paraphrases)

### 4.2 LLM — Llama 3.1 8B via OpenRouter (`openrouter/free`)

- **Type** : Transformer décodeur seul (LLM génératif)
- **Fournisseur** : OpenRouter (agrégateur multi-modèles)
- **Modèle** : `openrouter/free` — routeur intelligent qui sélectionne automatiquement un modèle gratuit disponible (actuellement NVIDIA Nemotron, Qwen, ou Llama selon disponibilité)
- **Température** : 0.3 (faible, pour des réponses précises et reproductibles)
- **Max tokens** : 800 (suffisant pour un rapport structuré)
- **Contexte** : Variable selon le modèle choisi par le routeur (au moins 8K tokens)
- **Coût** : 0 € (routeur gratuit)
- **Utilisation** : Uniquement pour la phase de génération (phase 3). Le retrieval est 100% déterministe (TF-IDF).

### 4.3 Pourquoi TF-IDF plutôt qu'un embedding dense ?

| Critère | TF-IDF (choisi) | Embedding dense |
|---------|-----------------|-----------------|
| Dépendance externe | Aucune | Sentence-transformers / API |
| Taille du modèle | 0 Ko (code natif) | ~500 Mo – 1 Go |
| Temps d'indexation | < 1 ms pour 90 docs | ~secondes |
| Explicabilité | Score traçable terme à terme | Boîte noire |
| Recherche par mot-clé | Excellente | Faible (dépend du contexte) |
| Recherche sémantique | Faible | Bonne |
| Maintenance | Aucune | Mise à jour des embeddings |

Pour un corpus de 90 documents réglementaires courts (< 200 mots chacun) et très spécialisés, le TF-IDF est plus adapté qu'un embedding dense : les termes juridiques sont précis et la recherche par mot-clé exacte est préférable à une approximation sémantique.

---

## 5. Base de connaissances

La base de connaissances contient **90 entrées** réparties en 4 zones :

### 5.1 Réglementation SaaS

| Zone | Entrées | Sources |
|------|---------|---------|
| 🇲🇦 Maroc — CNDP | 3 | Loi 09-08, Art. 3, 18, obligations de sécurité |
| 🇪🇺 UE — RGPD | 3 | Règlement 2016/679, Art. 5, 28, 46 |
| 🌍 International — SLA | 2 | Bonnes pratiques disponibilité et réversibilité |
| 🌍 International — Sécurité | 1 | Certifications ISO 27001, SOC 2, HDS, PCI-DSS |

### 5.2 Taxonomie Bâtiment (81 entrées chargées depuis `Taxonomie_V-1.xlsx`)

| Domaine | Sous-domaines | Nb d'entrées |
|---------|---------------|:------------:|
| **Habitation** | Individuelle (≤R+1, >R+1, RDC, R+1, >R+1), Collective (≤R+3, ≤R+7, grande hauteur), IGH (GHA > 50 m), ITGH > 200 m | 12 |
| **Établissements Recevant du Public (ERP)** | Types J à Y (accueil, conférence, commerce, restauration, hôtel, danse, enseignement, bibliothèque, exposition, santé, culte, administration, sport, musée), Types spéciaux (PA, CTS, SG, OA, PS, GA, EF, BM), Catégories 1 à 5, IGH (GHO, GHR, GHS, GHU, GHW, GHZ, GHTC, ITGH) | 36 |
| **Lieux de travail (ERT)** | Bureaux, Ateliers, Entrepôts | 3 |
| **Établissements Classés (EIC)** | Industries à risque, Entrepôts classés | 2 |
| **Glossaire** | BH, ERP, ERT, EIC, IGH, ITGH, GHA, GHO, GHR, GHS, GHU, GHW, GHZ, GHTC, PBDN | 30 |

### 5.3 Format des entrées

Chaque entrée de la base est un dictionnaire avec 4 champs :

```python
{
    "id": "taxo_12",           # Identifiant unique
    "source": "Taxonomie Bâtiment V1 — Établissements recevant du public (ERP)",
    "zone": "Réglementation Bâtiment",  # Filtre de zone
    "text": "1er groupe — ERP classiques — Type J – Structures d'accueil ... — Effectif total ≥ 100"
}
```

---

## 6. Intégration de la Taxonomie Bâtiment

### 6.1 Chargement depuis Excel

La fonction `_load_taxonomy_base()` dans `rag_conformite.py` :

1. Ouvre le fichier `Taxonomie_V-1.xlsx` avec `pandas.ExcelFile`
2. Parcourt la feuille **"Taxonomie"** (55 lignes) :
   - Colonne A (Domaine) → source (`"Taxonomie Bâtiment V1 — [Domaine]"`)
   - Colonnes B, C, D → concaténées dans `text` (séparateur `" — "`)
   - Les lignes vides (en-têtes de section) sont ignorées
3. Parcourt la feuille **"Glossaire"** (33 lignes) :
   - Colonne A (Acronyme) + Colonne B (Domaine) + Colonne C (Définition) → texte complet
   - Les définitions vides sont ignorées
4. Les 81 entrées produites sont fusionnées dans `KNOWLEDGE_BASE` via `KNOWLEDGE_BASE.extend()`

### 6.2 Indexation automatique

Lors de l'instanciation de `ConformiteRAG`, l'index TF-IDF est reconstruit sur l'ensemble des 90 documents. Les entrées de la taxonomie sont donc automatiquement indexées au même titre que les documents juridiques SaaS. Aucune configuration supplémentaire n'est nécessaire.

### 6.3 Fonction de classification dédiée

`classify_building(query, top_k=5)` :
- Utilise le même moteur de retrieval que la conformité
- Filtre uniquement sur la zone `"Réglementation Bâtiment"`
- Retourne les résultats structurés avec : score, sous-domaine, catégorie, caractéristiques, source

### 6.4 Fonction d'arbre ontologique

`get_taxonomy_tree()` :
- Parcourt les entrées `taxo_*` de la base
- Reconstruit la hiérarchie : **Domaine → Sous-domaine → [{Catégorie, Caractéristiques}]**
- Retourne un JSON structuré exploitable par l'interface web

---

## 7. Fonctionnalités dérivées — 3 modes d'exploitation

### 7.1 Agent conversationnel dédié

- **Déclencheur** : Sélection de l'agent "Taxonomie Bâtiment" dans l'interface
- **Prompt système** (1454 caractères) : Instruction spécialisée qui décrit les 4 domaines, les seuils (BH 1ère famille ≤ R+1, ERP 1ère catégorie > 1500 pers, IGH > 28 m, etc.)
- **Comportement** : L'agent pose des questions ciblées pour déterminer l'usage, le nombre d'étages, l'effectif et la hauteur, puis croise avec les seuils de la taxonomie
- **Résultat** : Classification complète avec sources réglementaires

### 7.2 Classification intelligente (endpoint REST)

- **Route** : `POST /api/taxonomie/classify`
- **Entrée** : Description textuelle du bâtiment
- **Traitement** : Vectorisation TF-IDF → similarité cosinus → top-5 entrées de la zone "Réglementation Bâtiment"
- **Sortie** : Tableau des classifications avec score de pertinence (0-1), catégorie, caractéristiques et source
- **Pas de LLM** : Résultat purement déterministe

### 7.3 Explorateur de l'arbre taxonomique (endpoint REST)

- **Route** : `GET /api/taxonomie/tree`
- **Traitement** : Parcours des entrées et reconstruction de l'arbre hiérarchique
- **Sortie** : JSON structuré avec 4 domaines, 51 entrées classifiées
- **Utilisation** : Chargé dans l'interface web sous forme de tree view interactive

---

## 8. Limites et perspectives

### 8.1 Limites actuelles

1. **TF-IDF vs embeddings sémantiques** : Un bâtiment décrit comme "petit immeuble de 4 niveaux" ne matchera pas parfaitement une entrée contenant "≤ R+3" car les termes sont différents. Une amélioration possible serait d'ajouter un embedding dense (via sentence-transformers) en parallèle du TF-IDF.

2. **Corpus statique** : La taxonomie est chargée depuis un fichier Excel fixe. Toute mise à jour nécessite une modification du fichier et un redémarrage de l'application.

3. **Pas de mise à jour incrémentale** : L'index TF-IDF est reconstruit intégralement à chaque démarrage. Pour 90 documents, le coût est négligeable, mais passerait mal à l'échelle (> 10 000 documents).

4. **Pas de pondération des champs** : Les champs Domaine, Sous-domaine, Catégorie et Caractéristiques sont concaténés sans pondération différentielle. Un terme dans le Domaine (ex: "Habitation") a le même poids que dans les Caractéristiques.

### 8.2 Perspectives d'amélioration

1. **Embeddings hybrides** : Combiner TF-IDF (précision des mots-clés) + embeddings denses (compréhension sémantique) avec un score pondéré.

2. **Taxonomie multi-source** : Charger dynamiquement plusieurs fichiers Excel ou API pour enrichir la base sans modification du code.

3. **Interface de visualisation** : Ajouter un graphe interactif (D3.js) de l'arbre ontologique pour naviguer visuellement entre les domaines, sous-domaines et catégories.

4. **Suggestions automatiques** : À partir d'une description partielle, proposer les classifications les plus probables avant même la recherche complète.

---

*Document généré le 13 juin 2026 — RAGulatory v1.0*
