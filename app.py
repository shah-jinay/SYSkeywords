from flask import Flask, render_template, request, jsonify
from extractor import extract

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract_keywords():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 20_000:
        return jsonify({"error": "Text too long (max 20 000 chars)"}), 400
    results = extract(text)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
