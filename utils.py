# utils.py
import datetime

# ==============================
# Fonctions utilitaires
# ==============================

def log_message(role: str, message: str):
    """
    Enregistre un message avec horodatage.
    role : 'user' ou 'chatbot'
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {role.upper()}: {message}")

def format_response(response: str) -> str:
    """
    Formate la réponse du chatbot :
    - Supprime espaces inutiles
    - Ajoute une mise en forme simple
    """
    response = response.strip()
    return f"> {response}"

def clean_text(text: str) -> str:
    """
    Nettoie un texte :
    - Supprime espaces multiples
    - Normalise Oui/Non
    """
    text = " ".join(text.split())
    text = text.replace("oui", "Oui").replace("non", "Non")
    return text

def validate_input(user_input: str) -> bool:
    """
    Vérifie si l'entrée utilisateur est valide.
    Retourne False si vide ou trop courte.
    """
    if not user_input or len(user_input.strip()) < 2:
        return False
    return True

# ==============================
# Exemple d’utilisation
# ==============================
if __name__ == "__main__":
    log_message("user", "Quels SaaS offrent un plan gratuit ?")
    response = "Voici les logiciels SaaS avec plan gratuit : Zoho Invoice, Wave Invoicing."
    print(format_response(response))
