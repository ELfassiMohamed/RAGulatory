# main.py
import os
import pandas as pd
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME, OPENROUTER_BASE_URL, MODEL_SETTINGS, DATASET_PATH, check_config
from dataset_loader import load_dataset, clean_dataset, filter_by_column
from prompt_builder import build_prompt
from utils import log_message, format_response, validate_input


check_config()  # Vérifie la config avant de lancer le chatbot

# 1. Charger la clé API depuis l'environnement
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# 2. Charger le dataset SaaS
df = pd.read_excel(DATASET_PATH)

# 3. Fonction principale du chatbot
def chatbot(question: str):
    prompt = build_prompt(question)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Tu es un expert du marché SaaS (Software as a Service). Ton rôle est de conseiller et de répondre aux questions sur le marché SaaS en utilisant les données fournies dans le contexte."},
            {"role": "user", "content": prompt}
        ],
        temperature=MODEL_SETTINGS["temperature"],
        max_tokens=MODEL_SETTINGS["max_tokens"]
    )
    return response.choices[0].message.content

# 4. Boucle d'interaction
if __name__ == "__main__":
    print("=== Chatbot SaaS (OpenRouter) ===")
    while True:
        user_input = input("Vous: ")
        if user_input.lower() in ["quit", "exit", "stop"]:
            print("Chatbot: À bientôt !")
            break
        answer = chatbot(user_input)
        print("Chatbot:", answer)
