"""
Customer Segmentation - Flask Backend (Advanced)
Loads model.pkl, scaler.pkl, and centroids.json.
Serves index.html and provides /predict and /api/predict endpoints.
Maps cluster index to label, color, description, and centroid values.
"""

import json
import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the trained model, scaler, and centroids once at startup
try:
    with open("model.pkl", "rb") as f:
        MODEL = pickle.load(f)
    print("✅ Model loaded successfully")

    with open("scaler.pkl", "rb") as f:
        SCALER = pickle.load(f)
    print("✅ Scaler loaded successfully")

    with open("centroids.json", "r") as f:
        CENTROIDS = json.load(f)
    print("✅ Centroids loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}")
    print("Please run 'python train_model.py' first to generate required files.")
    MODEL = None
    SCALER = None
    CENTROIDS = None

# Cluster index -> (label, color, description)
CLUSTER_MAP = {
    0: ("Low Income – Low Spending", "Gray", "Low Priority Customers"),
    1: ("High Income – High Spending", "Green", "Target Customers 💰"),
    2: ("Low Income – High Spending", "Orange", "Impulse Buyers"),
    3: ("High Income – Low Spending", "Red", "Potential Customers"),
    4: ("Average Customers", "Blue", "Moderate Value Customers"),
}

# Updated cluster meaning mapping (as per requirements)
CLUSTER_MEANINGS = {
    0: "Low Priority",
    1: "Target Customers",
    2: "Impulse Buyers",
    3: "Potential Customers",
    4: "Moderate Value",
}


def validate_input(income_str, score_str):
    """Validate income and spending score. Returns (ok, error_message)."""
    if not income_str or not score_str:
        return False, "Please fill in both Annual Income and Spending Score."
    income_str = income_str.strip()
    score_str = score_str.strip()
    if not income_str or not score_str:
        return False, "Please fill in both Annual Income and Spending Score."
    try:
        income = float(income_str)
    except ValueError:
        return False, "Annual Income must be a valid number."
    try:
        score = float(score_str)
    except ValueError:
        return False, "Spending Score must be a valid number."
    if income < 0:
        return False, "Annual Income cannot be negative."
    if score < 1 or score > 100:
        return False, "Spending Score must be between 1 and 100."
    return True, None


def predict_cluster(income, score):
    """
    Predict cluster for given income and score.
    Returns cluster_id, label, color, description, centroid_income, centroid_score.
    """
    if MODEL is None or SCALER is None or CENTROIDS is None:
        raise RuntimeError("Model, scaler, or centroids not loaded. Please train the model first.")

    # Scale input
    X = np.array([[income, score]])
    X_scaled = SCALER.transform(X)

    # Predict cluster
    cluster_id = int(MODEL.predict(X_scaled)[0])

    # Get cluster info
    label, color, description = CLUSTER_MAP[cluster_id]

    # Get centroid values
    centroid_data = CENTROIDS[str(cluster_id)]
    centroid_income = centroid_data["income"]
    centroid_score = centroid_data["score"]

    return cluster_id, label, color, description, centroid_income, centroid_score


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept income and score, return cluster label, color, description, and centroid values."""
    data = request.get_json() or {}
    income_str = data.get("income", "")
    score_str = data.get("score", "")

    ok, err = validate_input(income_str, score_str)
    if not ok:
        return jsonify({"success": False, "error": err}), 400

    try:
        income = float(income_str)
        score = float(score_str)

        _, label, color, description, centroid_income, centroid_score = predict_cluster(income, score)

        return jsonify({
            "success": True,
            "label": label,
            "color": color,
            "description": description,
            "centroid_income": round(centroid_income, 2),
            "centroid_score": round(centroid_score, 2),
        })
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for prediction.
    Accepts JSON: {"income": 70, "score": 80}
    Returns JSON with segment, centroid_income, centroid_score.
    """
    data = request.get_json() or {}

    # Handle both string and numeric inputs
    income_raw = data.get("income")
    score_raw = data.get("score")

    if income_raw is None or score_raw is None:
        return jsonify({"error": "Missing 'income' or 'score' in request body"}), 400

    # Convert to strings for validation
    income_str = str(income_raw)
    score_str = str(score_raw)

    ok, err = validate_input(income_str, score_str)
    if not ok:
        return jsonify({"error": err}), 400

    try:
        income = float(income_str)
        score = float(score_str)

        _, label, color, description, centroid_income, centroid_score = predict_cluster(income, score)

        return jsonify({
            "segment": label,
            "centroid_income": round(centroid_income, 2),
            "centroid_score": round(centroid_score, 2),
            "color": color,
            "description": description,
        })
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
