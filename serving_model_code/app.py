from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import io
from datetime import datetime
from google.cloud import storage

app = Flask(__name__)

# GCS path to the model
GCS_MODEL_PATH = "ai_ops_final_project25k/pipeline_root/256797333550/credit-card-fraud-ml-pipeline-20250415025856/train-xgboost-model_-8443954460404744192/model_output/model.pkl"

# Expected features
REQUIRED_FEATURES = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

# Download model from GCS if not already present
def download_model():
    local_path = "/tmp/model.pkl"
    if os.path.exists(local_path):
        return local_path

    print("🔄 Downloading model from GCS...")
    client = storage.Client()
    bucket = client.bucket(GCS_MODEL_PATH.split("/")[0])
    blob_path = "/".join(GCS_MODEL_PATH.split("/")[1:])
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print("✅ Model downloaded.")
    return local_path

# Load the model at startup
MODEL_PATH = download_model()
MODEL = pickle.load(open(MODEL_PATH, "rb"))
MODEL_VERSION = os.path.basename(MODEL_PATH)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "OK",
        "message": "Model is live on Cloud Run ✅",
        "model_version": MODEL_VERSION
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check content-type
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415

        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "Empty or invalid JSON received"}), 400

        df = pd.DataFrame(input_data)

        # Validate required features
        missing = [col for col in REQUIRED_FEATURES if col not in df.columns]
        if missing:
            return jsonify({"error": f"Missing required features: {missing}"}), 400

        
        if 'hour' not in df.columns:
            print("Adding required 'hour' feature that was missing")
            df['hour'] = 12  # Default mid-day value
        
        # Make predictions
        predictions = MODEL.predict(df)
        probabilities = MODEL.predict_proba(df)[:, 1]  # Get fraud probabilities
        
        result_df = df.copy()
        result_df["prediction"] = predictions
        result_df["fraud_probability"] = probabilities

        # Save to GCS
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        file_name = f"prediction_{now}.csv"
        bucket_name = "sample_request"
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        buffer = io.StringIO()
        result_df.to_csv(buffer, index=False)
        blob.upload_from_string(buffer.getvalue(), content_type="text/csv")

        # Log summary to Cloud Logging
        print(f"✅ Predictions made at {now} | Records: {len(df)} | Saved to: gs://{bucket_name}/{file_name}")

        return jsonify({
            "predictions": predictions.tolist(),
            "fraud_probabilities": probabilities.tolist(),
            "saved_to": f"gs://{bucket_name}/{file_name}",
            "model_version": MODEL_VERSION
        })

    except Exception as e:
        import traceback
        print("🔥 Error in /predict:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)