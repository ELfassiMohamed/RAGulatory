# web_app/app.py
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/taxonomie/recommend", methods=["POST"])
def taxonomie_recommend():
    from taxonomie_advisor import recommend as _recommend
    data = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Description du terrain requise"}), 400
    try:
        result = _recommend(query, method=data.get("method", "wsm"))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Warmup ML en arriere-plan
import threading
def _warmup():
    try:
        from ml_predictor import get_predictor
        get_predictor()
    except Exception:
        pass
threading.Thread(target=_warmup, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=True)
