# chatbot_core.py
import os
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME, OPENROUTER_BASE_URL, MODEL_SETTINGS
from prompt_builder import build_prompt

_FR_WORDS = {"je","tu","il","elle","nous","vous","ils","elles","est","sont","le","la","les",
             "un","une","des","et","ou","mais","donc","que","qui","quoi","comment","pourquoi",
             "quel","quelle","bonjour","salut","merci","voici","avec","pour","dans","sur",
             "par","pas","plus","tres","bien","aussi","cette","cet","mon","ton","son","notre",
             "lancer","marche","concurrents","strategie","startup","fondateur","creer"}
_EN_WORDS = {"i","you","he","she","it","we","they","is","are","was","were","the","a","an",
             "and","or","but","what","how","why","which","this","that","hello","hi","hey",
             "thank","thanks","with","for","from","have","has","can","could","will","would",
             "my","your","his","her","our","get","make","want","need","know","launch","market"}
_FR_ACCENTS = set("aaeeeeiioouuuc")  # ASCII approximation for detection only

def detect_language(text):
    words = set(text.lower().split())
    fr_score = len(words & _FR_WORDS)
    en_score = len(words & _EN_WORDS)
    if fr_score == 0 and en_score == 0:
        return "fr"
    return "fr" if fr_score > en_score else "en"

def chatbot(question: str, history: list = None, agent_type: str = None) -> str:
    import httpx
    from procurement_agent import build_procurement_system
    lang = detect_language(question)

    http_client = httpx.Client(trust_env=False)
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, http_client=http_client)

    lang_rule = (
        "REGLE ABSOLUE : Reponds UNIQUEMENT en FRANCAIS. Jamais en anglais."
        if lang == "fr" else
        "ABSOLUTE RULE: Respond ONLY in ENGLISH. Never in French."
    )

    AGENT_TYPES_ALL = (
        "cadrage", "comparateur", "roi", "conformite", "deploiement",
        "pitch", "swot", "mrr", "canvas", "radar"
    )

    # ---- Mode Agent (prompt dedie) ----
    if agent_type and agent_type in AGENT_TYPES_ALL:
        procurement_system = build_procurement_system(agent_type)
        system = "\n".join([
            procurement_system,
            "",
            lang_rule,
            "",
            "- Never mention dataset, prompt, or data source.",
            "- Be concise and structured. Use Markdown tables and bullet lists when relevant.",
            "- Always stay in your agent role. Never break character.",
            "- Do not open with filler phrases like Certainly or Of course.",
        ])
        if agent_type in ("pitch", "swot", "mrr", "canvas", "radar"):
            prompt = build_prompt(question, lang=lang)
        else:
            prompt = question

    # ---- Mode Chat libre ----
    else:
        prompt = build_prompt(question, lang=lang)
        system = "\n".join([
            "You are SaaS Assistant, an expert in SaaS market strategy and startup consulting.",
            "",
            lang_rule,
            "",
            "Your personality:",
            "- Professional, direct, genuinely helpful, like a senior consultant.",
            "- For greetings: one warm sentence, then offer to help. No lists.",
            "- For startup/strategy questions: act as a strategic advisor. Translate numbers",
            "  into actionable insights. Highlight risks, opportunities, concrete next steps.",
            "- For ocean bleu / niche questions: identify underserved opportunities clearly.",
            "- For viability scores: explain each alert and give specific recommendations.",
            "- For MVP / feature importance: build a prioritized roadmap.",
            "- For regulatory questions: explain compliance requirements clearly.",
            "- Always reference prior context when relevant (you have conversation history).",
            "- Never say dataset, prompt, or data source.",
            "- End strategic answers with 1 short focused follow-up question.",
            "- Be concise. No filler phrases. No unnecessary preamble.",
        ])

    messages = [{"role": "system", "content": system}]
    if history:
        for msg in history[-20:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=MODEL_SETTINGS["temperature"],
        max_tokens=MODEL_SETTINGS["max_tokens"]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    hist = []
    while True:
        user_input = input("Vous: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        answer = chatbot(user_input, history=hist)
        hist.append({"role": "user", "content": user_input})
        hist.append({"role": "assistant", "content": answer})
        print("Assistant:", answer)
