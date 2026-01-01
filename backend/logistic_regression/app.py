from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS


# -----------------------------
# App initialization
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Load model & scaler
# -----------------------------
model = joblib.load("Loan_Approval_Logistic_Regression.pkl")
scaler = joblib.load("Loan_Approval_Logistic_Regression_Standard_Scaler.pkl")

# -----------------------------
# FEATURE ORDER (VERY IMPORTANT)
# -----------------------------
FEATURE_ORDER = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]

# -----------------------------
# Health check route
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "Loan Approval Logistic Regression API is running 🚀"


# -----------------------------
# Prediction route
# -----------------------------
@app.route("/predict-logistic", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # ✅ FIX: convert loan_term from months → years
        data["loan_term"] = data["loan_term"] / 12

        # -----------------------------
        # Extract input features (SAFE)
        # -----------------------------
        features = [data[col] for col in FEATURE_ORDER]

        # Convert to numpy 2D array (force float)
        input_array = np.array(features, dtype=float).reshape(1, -1)

        # -----------------------------
        # Scaling
        # -----------------------------
        input_scaled = scaler.transform(input_array)

        # 🔍 DEBUG PRINTS (as-is)
        print("RAW JSON:", data)
        print("FEATURE LIST:", features)
        print("SCALED INPUT:", input_scaled)
        print("PROBA:", model.predict_proba(input_scaled))

        # -----------------------------
        # Prediction
        # -----------------------------
        probability = model.predict_proba(input_scaled)[0][1]
        prediction = int(probability >= 0.7)  # threshold tuned

        # -----------------------------
        # Response
        # -----------------------------
        return jsonify({
            "loan_approved": bool(prediction),
            "approval_probability": round(float(probability), 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
