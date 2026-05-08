from flask import Flask, render_template, request, jsonify
from extractor import extract
from comparator import compare

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
        return jsonify({"error": "Text too long (max 20,000 chars)"}), 400
    return jsonify(extract(text))


@app.route("/compare", methods=["POST"])
def compare_resume():
    data        = request.get_json(force=True)
    jd_data     = data.get("jd_data")
    resume_text = (data.get("resume_text") or "").strip()
    if not jd_data:
        return jsonify({"error": "Extract the JD first, then compare."}), 400
    if not resume_text:
        return jsonify({"error": "No resume text provided."}), 400
    if len(resume_text) > 20_000:
        return jsonify({"error": "Resume too long (max 20,000 chars)."}), 400
    return jsonify(compare(jd_data, resume_text))


if __name__ == "__main__":
    app.run(debug=True, port=5050)
