# direct_test.py
import joblib
import numpy as np

print("="*60)
print("🔍 DIRECT MODEL TEST")
print("="*60)

# 1. Model load करो
try:
    model = joblib.load('copd_risk_model.pkl')
    print("✅ Model loaded successfully!")
    # Model load hone ke baad
    print("Keys inside the file:", model.keys()) 
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# 2. Same features jo frontend se bheje the
test_features = np.array([[
    55,    # age
    26,    # bmi
    2,     # mmrc
    20,    # packYears
    1,     # exacerbations
    0,     # occupationalExposure
    0.65,  # fev1Fvc
    70,    # fev1Percent
    92,    # oxygen
    150,   # eosinophil
    65,    # dlco
    120,   # aat
]])

print("\n📤 TEST FEATURES:")
print(test_features)

# 3. Model se prediction लो
print("\n🎯 MODEL PREDICTION:")
# Agar key ka naam 'model' hai to:
actual_model = model['model'] 
raw_prediction = actual_model.predict(test_features)[0]
print(f"Raw prediction: {raw_prediction}")

# 4. Risk score calculate करो
if raw_prediction <= 1:
    risk_score = raw_prediction * 100
else:
    risk_score = raw_prediction

print(f"Risk score: {risk_score:.1f}%")

# 5. Risk level decide करो
if risk_score >= 61:
    risk_level = 'High Risk'
elif risk_score >= 31:
    risk_level = 'Moderate Risk'
else:
    risk_level = 'Low Risk'

print(f"Risk level: {risk_level}")

# 6. Probability check (अगर हो तो)
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(test_features)[0]
    print(f"\n📊 Class probabilities: {probabilities}")
    print(f"Confidence: {max(probabilities)*100:.1f}%")

print("\n" + "="*60)