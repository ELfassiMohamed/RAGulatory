"""
logger.py — Observabilite et Tracabilite (standard DevOps/MLOps)
================================================================
Enregistre chaque interaction IA dans logs/app.log :
  - Appels LLM (Groq) : prompt, tokens, duree, statut
  - Appels ML (predictions) : features, resultats, duree
  - Appels RAG : query, chunks, sources
  - Erreurs et exceptions

Format : JSON Lines (une ligne JSON par evenement) pour faciliter
l'analyse avec des outils de monitoring (Grafana, Kibana, etc.)
"""

import os
import json
import time
import functools
import traceback
from datetime import datetime, timezone

# ── Configuration ──────────────────────────────────────────────────────────────
LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
MAX_LOG_BYTES = 5 * 1024 * 1024   # 5 MB — rotation au dela

os.makedirs(LOG_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Fonctions de base
# ══════════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _rotate_if_needed():
    """Rotation simple : renomme app.log en app.log.1 si trop grand."""
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > MAX_LOG_BYTES:
        backup = LOG_FILE + ".1"
        if os.path.exists(backup):
            os.remove(backup)
        os.rename(LOG_FILE, backup)


def _write(event: dict):
    """Ecrit un evenement JSON sur une ligne dans le fichier de log."""
    _rotate_if_needed()
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass   # Ne jamais planter l'app a cause du logging


# ══════════════════════════════════════════════════════════════════════════════
# API publique — fonctions de log par type d'evenement
# ══════════════════════════════════════════════════════════════════════════════

def log_llm_call(
    prompt_preview: str,
    response_preview: str,
    model: str,
    agent_type: str = None,
    tokens_used: int = None,
    duration_ms: float = None,
    error: str = None
):
    """Log un appel au LLM (Groq/OpenAI)."""
    _write({
        "ts":           _now_iso(),
        "event":        "llm_call",
        "model":        model,
        "agent_type":   agent_type,
        "prompt_chars": len(prompt_preview),
        "prompt_preview": prompt_preview[:120] + ("..." if len(prompt_preview) > 120 else ""),
        "response_preview": response_preview[:120] + ("..." if len(response_preview) > 120 else ""),
        "tokens_used":  tokens_used,
        "duration_ms":  round(duration_ms, 1) if duration_ms else None,
        "status":       "error" if error else "ok",
        "error":        error
    })


def log_ml_prediction(
    task: str,
    features_summary: dict,
    result: dict,
    duration_ms: float = None,
    error: str = None
):
    """Log une prediction ML (Random Forest)."""
    _write({
        "ts":           _now_iso(),
        "event":        "ml_prediction",
        "task":         task,
        "features":     {k: v for k, v in list(features_summary.items())[:10]},
        "predicted":    result.get("label") if not error else None,
        "confidence":   result.get("confidence") if not error else None,
        "duration_ms":  round(duration_ms, 1) if duration_ms else None,
        "status":       "error" if error else "ok",
        "error":        error
    })


def log_rag_query(
    query: str,
    zone: str,
    chunks_found: int,
    top_source: str = None,
    duration_ms: float = None,
    error: str = None
):
    """Log une requete RAG (retrieval + generation)."""
    _write({
        "ts":           _now_iso(),
        "event":        "rag_query",
        "query_preview": query[:100],
        "zone":         zone,
        "chunks_found": chunks_found,
        "top_source":   top_source,
        "duration_ms":  round(duration_ms, 1) if duration_ms else None,
        "status":       "error" if error else "ok",
        "error":        error
    })


def log_nlu_extract(
    text_preview: str,
    entities_found: list,
    confidence: int,
    duration_ms: float = None,
    error: str = None
):
    """Log une extraction NLU."""
    _write({
        "ts":           _now_iso(),
        "event":        "nlu_extract",
        "text_preview": text_preview[:100],
        "entities":     entities_found,
        "confidence":   confidence,
        "duration_ms":  round(duration_ms, 1) if duration_ms else None,
        "status":       "error" if error else "ok",
        "error":        error
    })


def log_audit(
    saas_name: str,
    url: str,
    features_scraped: dict,
    predictions: dict,
    duration_ms: float = None,
    error: str = None
):
    """Log un audit SaaS (Mode Audit)."""
    _write({
        "ts":          _now_iso(),
        "event":       "audit_saas",
        "saas_name":   saas_name,
        "url":         url,
        "fields_found": len(features_scraped),
        "predictions": {k: v.get("label") for k, v in predictions.items()} if predictions else {},
        "duration_ms": round(duration_ms, 1) if duration_ms else None,
        "status":      "error" if error else "ok",
        "error":       error
    })


def log_error(context: str, error: Exception):
    """Log une erreur non prevue."""
    _write({
        "ts":       _now_iso(),
        "event":    "error",
        "context":  context,
        "type":     type(error).__name__,
        "message":  str(error),
        "traceback": traceback.format_exc()[-500:]
    })


# ══════════════════════════════════════════════════════════════════════════════
# Decorateur de timing
# ══════════════════════════════════════════════════════════════════════════════

def timed(func):
    """Decorateur qui mesure la duree d'execution d'une fonction."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return result, elapsed_ms
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# Lecture des logs (pour le dashboard admin)
# ══════════════════════════════════════════════════════════════════════════════

def read_recent_logs(n: int = 50, event_type: str = None) -> list:
    """
    Lit les N dernieres lignes du fichier de log.

    Args:
        n          : nombre de lignes a retourner
        event_type : filtre optionnel ("llm_call", "ml_prediction", etc.)

    Returns:
        list de dicts, du plus recent au plus ancien
    """
    if not os.path.exists(LOG_FILE):
        return []

    events = []
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                    if event_type is None or evt.get("event") == event_type:
                        events.append(evt)
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []

    return events[-n:][::-1]   # n derniers, ordre inverse (plus recent en premier)


def get_stats() -> dict:
    """
    Calcule des statistiques de base a partir des logs.

    Returns:
        dict avec counts par type, taux d'erreur, duree moyenne LLM
    """
    all_events = read_recent_logs(n=1000)
    if not all_events:
        return {"total": 0}

    counts     = {}
    errors     = 0
    llm_times  = []
    ml_times   = []

    for evt in all_events:
        etype = evt.get("event", "unknown")
        counts[etype] = counts.get(etype, 0) + 1
        if evt.get("status") == "error":
            errors += 1
        if etype == "llm_call" and evt.get("duration_ms"):
            llm_times.append(evt["duration_ms"])
        if etype == "ml_prediction" and evt.get("duration_ms"):
            ml_times.append(evt["duration_ms"])

    return {
        "total":           len(all_events),
        "by_type":         counts,
        "error_count":     errors,
        "error_rate_pct":  round(errors / len(all_events) * 100, 1) if all_events else 0,
        "avg_llm_ms":      round(sum(llm_times) / len(llm_times), 0) if llm_times else None,
        "avg_ml_ms":       round(sum(ml_times) / len(ml_times), 0) if ml_times else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log_llm_call(
        prompt_preview="Quel est le prix ideal pour un SaaS CRM PME France ?",
        response_preview="Selon le marche, le prix median est de 29 EUR/mois...",
        model="llama-3.3-70b-versatile",
        agent_type="classe_prix",
        tokens_used=312,
        duration_ms=1420.5
    )
    log_ml_prediction(
        task="classe_prix",
        features_summary={"has_API": 1, "has_mobile": 0, "prix_max_eur": 89},
        result={"label": "Low", "confidence": 91.2},
        duration_ms=23.4
    )
    log_rag_query(
        query="Mon SaaS est heberge aux USA, conforme CNDP ?",
        zone="Maroc",
        chunks_found=3,
        top_source="CNDP Maroc -- Loi 09-08",
        duration_ms=8.1
    )
    log_error("test_context", ValueError("Erreur de test"))

    print("Logs ecrits dans:", LOG_FILE)
    stats = get_stats()
    print("Stats:", stats)
    recent = read_recent_logs(n=5)
    print(f"5 derniers evenements : {[e['event'] for e in recent]}")
