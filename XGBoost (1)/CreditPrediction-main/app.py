import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# -----------------
# Load artifacts
# -----------------
MODEL_DIR = "output_model"

model = joblib.load(os.path.join(MODEL_DIR, "xgb_final_model.joblib"))
preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
feature_names = pd.read_csv(os.path.join(MODEL_DIR, "feature_names.csv")).iloc[:, 0].tolist()

# -----------------
# Flask app
# -----------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form
            form_data = {feature: request.form.get(feature) for feature in feature_names}

            # Chuyển dữ liệu sang DataFrame
            X_input = pd.DataFrame([{
                k: float(v) if v not in [None, ""] else 0
                for k, v in form_data.items()
            }], columns=feature_names)

            # Tiền xử lý
            X_proc = preprocessor.transform(X_input)

            # Dự đoán
            proba = model.predict_proba(X_proc)[:, 1][0]
            prediction = "Tốt (0)" if proba < 0.5 else "Xấu (1)"
            probability = round(float(proba), 4)

        except Exception as e:
            error = str(e)

    return render_template("index.html",
                           feature_names=feature_names,
                           prediction=prediction,
                           probability=probability,
                           error=error,
                           form_data=form_data)


if __name__ == "__main__":
    app.run(debug=True)
