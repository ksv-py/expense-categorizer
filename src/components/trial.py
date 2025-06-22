import numpy as np
import pandas as pd
import dill
import os

# Load saved objects
def load_object(file_path):
    with open(file_path, 'rb') as file:
        return dill.load(file)

# Paths to artifacts
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")
LABEL_ENCODER_PATH = os.path.join("artifacts", "label_encoder.pkl")

# Load model, preprocessor, and encoder
model = load_object(MODEL_PATH)
preprocessor = load_object(PREPROCESSOR_PATH)
label_encoder = load_object(LABEL_ENCODER_PATH)

# ✅ Sample input (replace with your test input)
# Example to force "Travel" prediction
sample_data = pd.DataFrame([{
    'Amount': 299,
    'Mode': 'online',
    'Subcategory': 'noodles'
}])


# Preprocess the sample data
X_processed = preprocessor.transform(sample_data)

# Make prediction
y_pred = model.predict(X_processed)

# Decode label if label encoder was used
predicted_class = label_encoder.inverse_transform(y_pred.astype(int))

print(f"Predicted Category: {predicted_class[0]}")
