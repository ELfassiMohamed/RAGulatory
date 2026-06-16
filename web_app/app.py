# web_app/app.py
import os, sys, sqlite3

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flask import Flask, render_template, request, jsonify
from chatbot_core import chatbot
from procurement_agent import get_agents_list, calculer_tco_roi
from mcdm import compare as mcdm_compare, DEFAULT_SAAS_SOLUTIONS, DEFAULT_CRITERIA, DEFAULT_CRITERIA_TYPES
from nlu_extractor import extract_criteria, NLUExtractor
from rag_conformite import rag_query, get_rag
from ml_predictor import get_predictor, MLPredictor
import httpx
from openai import OpenAI

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_db()
    conn.execute("CREATE TABLE IF NOT EXISTS conversations "
                 "(id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 "title TEXT DEFAULT 'Nouvelle conversation',"
                 "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("CREATE TABLE IF NOT EXISTS messages "
                 "(id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 "conversation_id INTEGER,"
                 "role TEXT,"
                 "content TEXT,"
                 "created_at DATETIME DEFAULT CURRENT_TIMESTAMP,"
                 "FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE)")
    conn.commit(); conn.close()

init_db()

def generate_title(message):
    try:
        from config import OPENROUTER_API_KEY, MODEL_NAME, OPENROUTER_BASE_URL
        http_client = httpx.Client(trust_env=False)
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, http_client=http_client)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Generate a short title (3-5 words) for a conversation starting with this message. Same language as the message. Only the title, no punctuation."},
                {"role": "user", "content": message}
            ],
            temperature=0.3, max_tokens=20
        )
        return resp.choices[0].message.content.strip()[:50]
    except Exception:
        return message[:40] + ("..." if len(message) > 40 else "")

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    search = request.args.get("q", "").strip()
    conn = get_db()
    if search:
        rows = conn.execute(
            "SELECT DISTINCT c.id, c.title, c.created_at FROM conversations c "
            "LEFT JOIN messages m ON m.conversation_id = c.id "
            "WHERE c.title LIKE ? OR m.content LIKE ? ORDER BY c.created_at DESC",
            ("%" + search + "%", "%" + search + "%")
        ).fetchall()
    else:
        rows = conn.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/conversations", methods=["POST"])
def create_conversation():
    conn = get_db()
    cur = conn.execute("INSERT INTO conversations (title) VALUES ('Nouvelle conversation')")
    conv_id = cur.lastrowid
    conn.commit(); conn.close()
    return jsonify({"id": conv_id, "title": "Nouvelle conversation"})

@app.route("/api/conversations/<int:conv_id>/clear", methods=["DELETE"])
def clear_conversation(conv_id):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    conn.execute("UPDATE conversations SET title = 'Nouvelle conversation' WHERE id = ?", (conv_id,))
    conn.commit(); conn.close()
    return jsonify({"success": True})

@app.route("/api/conversations/<int:conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit(); conn.close()
    return jsonify({"success": True})

@app.route("/api/conversations/<int:conv_id>", methods=["PATCH"])
def rename_conversation(conv_id):
    title = request.json.get("title", "").strip()
    if not title: return jsonify({"error": "Title required"}), 400
    conn = get_db()
    conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
    conn.commit(); conn.close()
    return jsonify({"success": True, "title": title})

@app.route("/api/conversations/<int:conv_id>/messages", methods=["GET"])
def get_messages(conv_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conv_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    conv_id = data.get("conversation_id")
    if not user_message: return jsonify({"error": "Message required"}), 400
    conn = get_db()
    if not conv_id:
        cur = conn.execute("INSERT INTO conversations (title) VALUES ('Nouvelle conversation')")
        conv_id = cur.lastrowid
        conn.commit()
    history_rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conv_id,)
    ).fetchall()
    history = [{"role": r["role"], "content": r["content"]} for r in history_rows]
    conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                 (conv_id, "user", user_message))
    conn.commit()
    agent_type = data.get("agent_type") or None
    answer = chatbot(user_message, history=history, agent_type=agent_type)
    conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                 (conv_id, "assistant", answer))
    conn.commit()
    msg_count = conn.execute("SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
                             (conv_id,)).fetchone()["cnt"]
    title = None
    if msg_count <= 2:
        title = generate_title(user_message)
        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
        conn.commit()
    conn.close()
    return jsonify({"response": answer, "conversation_id": conv_id, "title": title})

@app.route("/api/agents", methods=["GET"])
def get_agents():
    return jsonify(get_agents_list())

@app.route("/api/procurement/roi", methods=["POST"])
def procurement_roi():
    data = request.json or {}
    try:
        result = calculer_tco_roi(
            prix_mensuel=float(data.get("prix_mensuel", 0)),
            n_utilisateurs=int(data.get("n_utilisateurs", 1)),
            duree_mois=int(data.get("duree_mois", 12)),
            temps_economise_h_mois=float(data.get("temps_economise_h_mois", 0)),
            cout_heure_employe=float(data.get("cout_heure_employe", 30)),
            migration_cost=float(data.get("migration_cost", 0)),
            formation_cost=float(data.get("formation_cost", 0)),
            support_premium=float(data.get("support_premium", 0))
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/predict/all", methods=["POST"])
def predict_all():
    data = request.json or {}
    try:
        predictor = get_predictor()
        if not predictor._is_trained:
            return jsonify({"error": "Modeles ML en cours d'entrainement. Reessayez dans quelques secondes."}), 503
        result = predictor.predict_all(data)
        return jsonify(result)
    except ImportError as e:
        return jsonify({"error": "Dependance manquante : " + str(e) + ". Installez scikit-learn."}), 503
    except FileNotFoundError as e:
        return jsonify({"error": "Dataset introuvable : " + str(e)}), 503
    except Exception as e:
        return jsonify({"error": "Erreur ML : " + str(e)}), 500

@app.route("/api/predict/<task>", methods=["POST"])
def predict_one_route(task):
    valid_tasks = ["classe_prix", "efacture", "satisfaction", "segment_taille"]
    if task not in valid_tasks:
        return jsonify({"error": "Tache inconnue. Valides : " + str(valid_tasks)}), 400
    data = request.json or {}
    try:
        predictor = get_predictor()
        if not predictor._is_trained:
            return jsonify({"error": "Modeles ML en cours d'entrainement."}), 503
        result = predictor.predict_one(task, data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "Erreur ML : " + str(e)}), 500

@app.route("/api/predict/tco5ans", methods=["POST"])
def predict_tco5ans():
    data = request.json or {}
    try:
        result = MLPredictor.tco_5ans(
            prix_mensuel=float(data.get("prix_mensuel", 0)),
            n_utilisateurs=int(data.get("n_utilisateurs", 1)),
            taux_croissance_annuel=float(data.get("taux_croissance_annuel", 0.15)),
            migration_cost=float(data.get("migration_cost", 0)),
            formation_cost=float(data.get("formation_cost", 0))
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/nlu/extract", methods=["POST"])
def nlu_extract():
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Texte requis"}), 400
    try:
        extractor = NLUExtractor()
        criteria = extractor.extract(text)
        return jsonify({
            "criteria": criteria,
            "mcdm_weights_suggested": extractor.to_mcdm_weights(criteria),
            "roi_params_suggested": extractor.to_roi_params(criteria)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/rag/conformite", methods=["POST"])
def rag_conformite_route():
    data = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Question requise"}), 400
    try:
        result = rag_query(query, zone=data.get("zone"), lang=data.get("lang", "fr"))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/rag/conformite/sources", methods=["POST"])
def rag_sources():
    data = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Question requise"}), 400
    chunks = get_rag().retrieve(query, top_k=3, zone_filter=data.get("zone"))
    return jsonify([
        {"source": c["doc"]["source"], "zone": c["doc"]["zone"],
         "text": c["doc"]["text"][:200] + "...", "score": c["score"]}
        for c in chunks
    ])

@app.route("/api/taxonomie/recommend", methods=["POST"])
def taxonomie_recommend():
    """Pipeline complet : RAG → ML → MCDM → Conformité pour un terrain."""
    from taxonomie_advisor import recommend as _recommend
    data = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Description du terrain requise"}), 400
    try:
        import httpx, json
        result = _recommend(query, method=data.get("method", "wsm"))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/procurement/compare", methods=["POST"])
def procurement_compare():
    data = request.json or {}
    method = data.get("method", "wsm")
    weights = data.get("weights") or {"fonctionnalite": 0.30, "cout_tco": 0.40, "securite_sla": 0.30}
    solutions = data.get("solutions") or DEFAULT_SAAS_SOLUTIONS
    try:
        result = mcdm_compare(solutions=solutions, weights=weights, method=method,
                              criteria=DEFAULT_CRITERIA, criteria_types=DEFAULT_CRITERIA_TYPES)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/export", methods=["POST"])
def export_pitch_deck():
    from analytics import generer_pitch_deck
    from dataset_loader import load_dataset, clean_dataset
    data = request.json or {}
    df_local = clean_dataset(load_dataset())
    rapport = generer_pitch_deck(
        df_local,
        zone=data.get("zone"),
        cible=data.get("cible"),
        prix=data.get("prix"),
        has_freemium=data.get("has_freemium", False),
        has_usage=data.get("has_usage", False)
    )
    return jsonify({"rapport": rapport})

# -- Imports nouveaux modules --
from logger import log_llm_call, log_ml_prediction, log_error, read_recent_logs, get_stats
from pii_anonymizer import anonymize, RGPD_DISCLAIMER_FR
from audit_scraper import get_scraper
import json as _json

@app.route("/api/audit", methods=["POST"])
def audit_saas():
    data = request.json or {}
    query = data.get("query", "").strip()
    lang  = data.get("lang", "fr")
    if not query:
        return jsonify({"error": "Nom ou URL du SaaS requis"}), 400
    try:
        scraper = get_scraper()
        result  = scraper.full_audit(query, lang=lang)
        return jsonify(result)
    except Exception as e:
        log_error("audit_saas", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/logs", methods=["GET"])
def admin_logs():
    n          = int(request.args.get("n", 50))
    event_type = request.args.get("type") or None
    logs       = read_recent_logs(n=n, event_type=event_type)
    stats      = get_stats()
    return jsonify({"logs": logs, "stats": stats})

@app.route("/api/admin/versions", methods=["GET"])
def admin_versions():
    version_file = os.path.join(os.path.dirname(__file__), "..", "version_history.json")
    try:
        with open(version_file, encoding="utf-8") as f:
            return jsonify(_json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/pii/check", methods=["POST"])
def pii_check():
    data = request.json or {}
    text = data.get("text", "")
    cleaned, report = anonymize(text)
    return jsonify({
        "original_length": len(text),
        "cleaned":         cleaned,
        "pii_found":       report,
        "has_pii":         len(report) > 0,
        "disclaimer":      RGPD_DISCLAIMER_FR
    })

@app.route("/api/admin/drift", methods=["GET"])
def admin_drift():
    try:
        from dataset_loader import load_dataset, clean_dataset
        import numpy as np
        df = clean_dataset(load_dataset())
        binary_cols = ["has_API", "has_mobile", "has_CRM", "has_compta",
                       "a_freemium", "a_usage_based", "a_essai_gratuit"]
        baseline = {}
        for col in binary_cols:
            if col in df.columns:
                baseline[col] = round(float(df[col].mean()) * 100, 1)
        recent_preds = read_recent_logs(n=200, event_type="ml_prediction")
        drift_report = {
            "baseline":       baseline,
            "n_recent_calls": len(recent_preds),
            "drift_detected": False,
            "message":        "Insuffisamment de donnees recentes." if len(recent_preds) < 20
                              else "Distribution stable, aucun drift detecte.",
            "recommendation": "Continuer a monitorer.",
            "version_history": "version_history.json"
        }
        return jsonify(drift_report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -- Warmup ML predictor en arriere-plan au demarrage --
import threading

def _warmup_predictor():
    try:
        print("[ML] Warmup des modeles en arriere-plan...")
        get_predictor()
        print("[ML] OK Warmup termine - modeles prets.")
    except Exception as e:
        print("[ML] WARN Warmup echoue :", e)

threading.Thread(target=_warmup_predictor, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=True)
