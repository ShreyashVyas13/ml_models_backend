# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import pandas as pd
# import os

# app = Flask(__name__)
# CORS(app)

# # 🔹 Load trained model
# model_info = joblib.load("SLR_Expense_Tracker_Model_V1.pkl")
# model = model_info["Expense_Tracker"]

# @app.route("/", methods=["GET"])
# def home():
#     return "Monthly Expense Prediction API is running"

# # @app.route("/predict", methods=["POST"])
# # def predict():
# #     data = request.get_json()
    
# #     income = data.get("income")

# #     # 🔒 Safety check
# #     if income is None:
# #         return jsonify({"error": "Income is required"}), 400

# #     # Model expects DataFrame
# #     input_df = pd.DataFrame({"Income": [income]})

# #     prediction = model.predict(input_df)[0]
# #     prediction = max(prediction, income * 0.4)

# #     return jsonify({
# #         "income": income,
# #         "predicted_expense": round(float(prediction), 2)
# #     })
# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     income = data.get("income")

#     if income is None or income <= 0:
#         return jsonify({"error": "Valid income is required"}), 400

#     input_df = pd.DataFrame({"Income": [income]})

#     prediction = model.predict(input_df)[0]

#     # ✅ Business rule clamp
#     min_expense = income * 0.4   # minimum 40% of income
#     prediction = max(prediction, min_expense)

#     return jsonify({
#         "income": income,
#         "predicted_expense": round(float(prediction), 2)
#     })

# if __name__ == "__main__":
#      port = int(os.environ.get("PORT", 5000))
#      app.run(host="0.0.0.0", port=port)


from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

# -----------------------------
# App initialization
# -----------------------------
app = Flask(__name__)
CORS(app)

# =====================================================
# 🔹 LOAD MODELS
# =====================================================

# Logistic Regression – Loan Approval
logistic_model = joblib.load("Loan_Approval_Logistic_Regression.pkl")
logistic_scaler = joblib.load("Loan_Approval_Logistic_Regression_Standard_Scaler.pkl")

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

# Linear Regression – Expense Tracker
linear_model_info = joblib.load("SLR_Expense_Tracker_Model_V1.pkl")
linear_model = linear_model_info["Expense_Tracker"]

# =====================================================
# 🔹 HEALTH CHECK
# =====================================================
@app.route("/", methods=["GET"])
def home():
    return "ML Models Backend is running 🚀"

# =====================================================
# 🔹 LOGISTIC REGRESSION ENDPOINT
# =====================================================
@app.route("/predict-logistic", methods=["POST"])
def predict_logistic():
    data = request.get_json()

    try:
        # convert loan_term from months → years
        data["loan_term"] = data["loan_term"] / 12

        features = [data[col] for col in FEATURE_ORDER]
        input_array = np.array(features, dtype=float).reshape(1, -1)
        input_scaled = logistic_scaler.transform(input_array)

        probability = logistic_model.predict_proba(input_scaled)[0][1]
        prediction = int(probability >= 0.7)

        return jsonify({
            "loan_approved": bool(prediction),
            "approval_probability": round(float(probability), 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# =====================================================
# 🔹 LINEAR REGRESSION ENDPOINT
# =====================================================
@app.route("/predict-linear", methods=["POST"])
def predict_linear():
    data = request.get_json()
    income = data.get("income")

    if income is None or income <= 0:
        return jsonify({"error": "Valid income is required"}), 400

    input_df = pd.DataFrame({"Income": [income]})
    prediction = linear_model.predict(input_df)[0]

    # Business rule clamp
    min_expense = income * 0.4
    prediction = max(prediction, min_expense)

    return jsonify({
        "income": income,
        "predicted_expense": round(float(prediction), 2)
    })

# =====================================================
# 🔹 RUN
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
