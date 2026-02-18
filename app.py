from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- YE IMPORT ZAROORI HAI
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # <-- YE LINE CONNECTION ENABLE KARTI HAI

# 1. Sahi Model File Load karein
# Dhyan rahe ye file 'app.py' ke saath same folder mein ho
try:
    bundle = joblib.load('copd_risk_model.pkl') # <-- Screenshot wala naam use kiya hai
    model = bundle['model']
    scaler = bundle['scaler']
    imputer = bundle['imputer']
    # Agar bundle mein feature names saved hain, wahi use karein safe side ke liye
    model_features = bundle.get('features', [
        'Age', 'BMI', 'Smoking_Pack_Years', 'FEV1_FVC_Ratio',
        'FEV1_Percent_Predicted', 'Oxygen_Saturation_SpO2', 'mMRC_Scale',
        'Exacerbations_History', 'Occupational_Exposure', 'Eosinophil_Count',
        'DLCO_Level', 'AAT_Level'
    ])
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Check if 'copd_risk_model.pkl' exists in the folder.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)  # Debugging ke liye

        # 2. Data Mapping (Frontend names -> Model names)
        input_dict = {
            'Age': data.get('age'),
            'BMI': data.get('bmi'),
            'Smoking_Pack_Years': data.get('packYears'),
            'FEV1_FVC_Ratio': data.get('fev1Fvc'),
            'FEV1_Percent_Predicted': data.get('fev1'),
            'Oxygen_Saturation_SpO2': data.get('oxygen'),
            'mMRC_Scale': data.get('mmrc'),
            'Exacerbations_History': data.get('exacerbations'),
            'Occupational_Exposure': data.get('occupationalExposure'),
            'Eosinophil_Count': data.get('eosinophil'),
            'DLCO_Level': data.get('dlco'),
            'AAT_Level': data.get('aat')
        }

        # Handle Missing Values (None -> NaN)
        for k, v in input_dict.items():
            if v is None:
                input_dict[k] = np.nan

        # 3. Create DataFrame with Correct Column Order
        df = pd.DataFrame([input_dict], columns=model_features)

        # 4. Preprocessing
        df_imputed = pd.DataFrame(imputer.transform(df), columns=model_features)
        df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=model_features)

        # 5. Prediction
        prob = model.predict_proba(df_scaled)[0][1]
        risk_score = round(prob * 100, 1)

        # Result Logic
        if prob > 0.7:
            result_text = "High Risk"
        elif prob > 0.3:
            result_text = "Moderate Risk"
        else:
            result_text = "Low Risk"

        return jsonify({
            'status': 'success',
            'risk_score': risk_score,
            'result': result_text
        })

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)