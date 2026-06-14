# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_NAME = "openrouter/free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_SETTINGS = {
    "temperature": 0.7,
    "max_tokens": 1024,
}

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_final.xlsx")

def check_config():
    if not OPENROUTER_API_KEY:
        raise ValueError("La cle API OpenRouter n'est pas definie. Creez un fichier .env avec OPENROUTER_API_KEY=votre_cle")
    print("Configuration chargee correctement.")
