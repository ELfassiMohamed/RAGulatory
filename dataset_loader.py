# dataset_loader.py
import pandas as pd
from config import DATASET_PATH

# ==============================
# Chargement et préparation du dataset SaaS
# ==============================

def load_dataset():
    """
    Charge le fichier Excel SaaS et retourne un DataFrame pandas.
    """
    try:
        df = pd.read_excel(DATASET_PATH)
        print(f"[OK] Dataset charge avec {len(df)} lignes et {len(df.columns)} colonnes.")
        return df
    except Exception as e:
        raise FileNotFoundError(f"[ERREUR] Impossible de charger le dataset : {e}")

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et prépare le dataset :
    - Supprime les doublons
    - Remplace les valeurs manquantes numériques par 0
    - Remplace les valeurs manquantes texte par 'Inconnu'
    """
    # Supprimer doublons
    df = df.drop_duplicates()

    # Remplacer valeurs manquantes
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("Inconnu")

    print("[OK] Dataset nettoye et pret.")
    return df

def filter_by_column(df: pd.DataFrame, column: str, value):
    """
    Filtre le dataset selon une colonne et une valeur donnée.
    Pour les colonnes binaires (0/1), utiliser 1 ou 0.
    Exemple : filter_by_column(df, "a_freemium", 1)
    """
    if column not in df.columns:
        raise ValueError(f"[ERREUR] La colonne '{column}' n'existe pas dans le dataset.")
    return df[df[column] == value]

def get_dataset_summary(df: pd.DataFrame) -> str:
    """
    Retourne un résumé textuel du dataset pour enrichir le contexte du chatbot.
    """
    summary_parts = []
    summary_parts.append(f"Le dataset contient {len(df)} logiciels SaaS.")

    # Zones géographiques
    zones = df['zone'].value_counts()
    summary_parts.append(f"Zones : {', '.join([f'{z} ({c})' for z, c in zones.items()])}")

    # Cibles
    cibles = df['cible'].value_counts()
    summary_parts.append(f"Cibles : {', '.join([f'{c} ({n})' for c, n in cibles.items()])}")

    # Prix
    summary_parts.append(f"Prix min moyen : {df['prix_min_eur'].mean():.2f}€, Prix max moyen : {df['prix_max_eur'].mean():.2f}€")

    # Fonctionnalités
    features = {
        'a_freemium': 'freemium',
        'a_essai_gratuit': 'essai gratuit',
        'has_API': 'API',
        'has_CRM': 'CRM',
        'has_compta': 'comptabilité',
        'has_mobile': 'mobile',
        'has_PDP': 'PDP',
        'conforme_e_facture_2026': 'e-facture 2026'
    }
    for col, label in features.items():
        count = df[df[col] == 1].shape[0]
        summary_parts.append(f"  - Avec {label} : {count}")

    return "\n".join(summary_parts)

def dataset_to_context(df: pd.DataFrame) -> str:
    """
    Convertit le dataset en un contexte textuel complet pour le LLM.
    Inclut les statistiques et un échantillon des données.
    """
    context_parts = []

    # Résumé global
    context_parts.append(get_dataset_summary(df))

    # Statistiques par zone
    context_parts.append("\n--- Statistiques par zone ---")
    for zone in df['zone'].unique():
        zone_df = df[df['zone'] == zone]
        context_parts.append(f"\n{zone} ({len(zone_df)} logiciels):")
        context_parts.append(f"  Prix min moyen: {zone_df['prix_min_eur'].mean():.2f}€")
        context_parts.append(f"  Note G2 moyenne: {zone_df['note_g2'].mean():.2f}")
        context_parts.append(f"  Satisfaction moyenne: {zone_df['satisfaction_pct'].mean():.1f}%")

    # Statistiques par classe de prix
    context_parts.append("\n--- Statistiques par classe de prix ---")
    for classe in df['Classe_prix'].unique():
        classe_df = df[df['Classe_prix'] == classe]
        context_parts.append(f"\n{classe} ({len(classe_df)} logiciels):")
        context_parts.append(f"  Prix min moyen: {classe_df['prix_min_eur'].mean():.2f}€")
        context_parts.append(f"  Prix max moyen: {classe_df['prix_max_eur'].mean():.2f}€")

    # Statistiques par segment de taille
    context_parts.append("\n--- Statistiques par segment de taille ---")
    for seg in df['segment_taille'].unique():
        seg_df = df[df['segment_taille'] == seg]
        context_parts.append(f"\n{seg} ({len(seg_df)} logiciels):")
        context_parts.append(f"  Note G2 moyenne: {seg_df['note_g2'].mean():.2f}")
        context_parts.append(f"  Nb employés moyen: {seg_df['nb_employes'].mean():.0f}")

    return "\n".join(context_parts)

# ==============================
# Exemple d'utilisation
# ==============================
if __name__ == "__main__":
    dataset = load_dataset()
    dataset = clean_dataset(dataset)

    # Afficher le résumé
    print(get_dataset_summary(dataset))

    # Exemple : filtrer les logiciels avec freemium
    freemium = filter_by_column(dataset, "a_freemium", 1)
    print(f"\nLogiciels avec freemium : {len(freemium)}")
