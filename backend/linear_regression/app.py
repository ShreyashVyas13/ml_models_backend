from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# 🔹 Load trained model
model_info = joblib.load("SLR_Expense_Tracker_Model_V1.pkl")
model = model_info["Expense_Tracker"]

@app.route("/", methods=["GET"])
def home():
    return "Monthly Expense Prediction API is running"

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
    
#     income = data.get("income")

#     # 🔒 Safety check
#     if income is None:
#         return jsonify({"error": "Income is required"}), 400

#     # Model expects DataFrame
#     input_df = pd.DataFrame({"Income": [income]})

#     prediction = model.predict(input_df)[0]
#     prediction = max(prediction, income * 0.4)

#     return jsonify({
#         "income": income,
#         "predicted_expense": round(float(prediction), 2)
#     })
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    income = data.get("income")

    if income is None or income <= 0:
        return jsonify({"error": "Valid income is required"}), 400

    input_df = pd.DataFrame({"Income": [income]})

    prediction = model.predict(input_df)[0]

    # ✅ Business rule clamp
    min_expense = income * 0.4   # minimum 40% of income
    prediction = max(prediction, min_expense)

    return jsonify({
        "income": income,
        "predicted_expense": round(float(prediction), 2)
    })

if __name__ == "__main__":
     port = int(os.environ.get("PORT", 5000))
     app.run(host="0.0.0.0", port=port)
