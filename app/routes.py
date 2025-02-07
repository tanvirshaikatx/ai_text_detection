from flask import Blueprint, render_template, request, jsonify
from app.model import predict

main = Blueprint("main", __name__)

@main.route("/")
def home():
    return render_template("index.html")

@main.route("/predict", methods=["POST"])
def predict_text():
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = predict(text)
    return jsonify({"prediction": result})
