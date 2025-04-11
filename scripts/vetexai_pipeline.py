# vertex_batch.py - Updated with Confusion Matrix Support
# train deploy predict with batch prediction

import pandas as pd
import numpy as np
import os
import datetime
import argparse
from kfp import dsl
from kfp import compiler
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from google.cloud import aiplatform

# Set your GCP project ID and bucket
PROJECT_ID = "ai-ops-class"
BUCKET_NAME = "ai_ops_final_project25k"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root/"
REGION = "us-central1"

# Data preprocessing component
@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas", 
        "scikit-learn", 
        "numpy", 
        "google-cloud-storage", 
        "fsspec", 
        "gcsfs"
    ],
)
def preprocess_data(
    data_path: str,
    processed_data_output: Output[Dataset]
):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, RobustScaler
    import os
    
    print(f"Reading data from: {data_path}")
    
    # Read the dataset
    df = pd.read_csv(data_path)
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    # Keep a copy of the original data with 'Time' column removed
    df_original = df.drop(['Time'], axis=1)
    
    # Handle missing values (if any)
    df.fillna(df.mean(), inplace=True)
    
    # Create feature 'hour' from 'Time'
    df['hour'] = df['Time'].apply(lambda x: np.floor(x / 3600))
    
    # Drop the 'Time' column
    df = df.drop(['Time'], axis=1)
    
    # Separate the features from the target variable
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    # Feature scaling using StandardScaler for amount column
    amount_scaler = StandardScaler()
    X['Amount'] = amount_scaler.fit_transform(X[['Amount']])
    
    # Robust scaling for other numerical features to handle outliers
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    num_cols = num_cols.drop('Amount')  # Exclude Amount as it's already scaled
    
    robust_scaler = RobustScaler()
    X[num_cols] = robust_scaler.fit_transform(X[num_cols])
    
    # Combine features and target into a processed dataframe
    processed_df = pd.concat([X, y], axis=1)
    
    # Save preprocessed data
    print(f"Saving processed data to: {processed_data_output.path}")
    processed_df.to_csv(processed_data_output.path, index=False)
    
    print(f"Data preprocessing completed. Processed data saved to {processed_data_output.path}")

# Data splitting component
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "numpy", "fsspec", "gcsfs"],
)
def split_data(
    processed_data_path: Input[Dataset],
    train_data: Output[Dataset],
    validation_data: Output[Dataset],
    test_data: Output[Dataset]
):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import os
    
    print(f"Loading preprocessed data from: {processed_data_path.path}")
    # Load the preprocessed data
    df = pd.read_csv(processed_data_path.path)
    print(f"Preprocessed data loaded successfully. Shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    # First split: training+validation vs test (80% vs 20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: training vs validation (75% vs 25% of the training+validation set)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    # Save the data splits
    print(f"Creating training dataset...")
    train_df = pd.concat([X_train, pd.DataFrame(y_train, columns=['Class'])], axis=1)
    print(f"Creating validation dataset...")
    val_df = pd.concat([X_val, pd.DataFrame(y_val, columns=['Class'])], axis=1)
    print(f"Creating test dataset...")
    test_df = pd.concat([X_test, pd.DataFrame(y_test, columns=['Class'])], axis=1)
    
    print(f"Saving training data to: {train_data.path}")
    train_df.to_csv(train_data.path, index=False)
    print(f"Saving validation data to: {validation_data.path}")
    val_df.to_csv(validation_data.path, index=False)
    print(f"Saving test data to: {test_data.path}")
    test_df.to_csv(test_data.path, index=False)
    
    print(f"Data split completed:")
    print(f"  - Training set size: {len(train_df)}")
    print(f"  - Validation set size: {len(val_df)}")
    print(f"  - Test set size: {len(test_df)}")

# Simple XGBoost model training component
@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "numpy",
        "seaborn",
        "fsspec",
        "gcsfs"
    ],
)
def train_xgboost_model(
    train_data: Input[Dataset],
    validation_data: Input[Dataset],
    model_output: Output[Model]
):
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import confusion_matrix
    import pickle
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import traceback
    
    try:
        print(f"Loading training data from: {train_data.path}")
        # Load training data
        train_df = pd.read_csv(train_data.path)
        print(f"Training data loaded successfully. Shape: {train_df.shape}")
        
        print(f"Loading validation data from: {validation_data.path}")
        # Load validation data
        val_df = pd.read_csv(validation_data.path)
        print(f"Validation data loaded successfully. Shape: {val_df.shape}")
        
        # Print column names to verify data structure
        print(f"Training data columns: {train_df.columns.tolist()}")
        print(f"Validation data columns: {val_df.columns.tolist()}")
        
        # Check for Class column in both datasets
        if 'Class' not in train_df.columns or 'Class' not in val_df.columns:
            print("Error: 'Class' column not found in data.")
            print(f"Available columns in training data: {train_df.columns.tolist()}")
            print(f"Available columns in validation data: {val_df.columns.tolist()}")
            raise ValueError("'Class' column not found in data.")
        
        # Split features and target
        X_train = train_df.drop(['Class'], axis=1)
        y_train = train_df['Class']
        X_val = val_df.drop(['Class'], axis=1)
        y_val = val_df['Class']
        
        print(f"Training features shape: {X_train.shape}, Training target shape: {y_train.shape}")
        print(f"Validation features shape: {X_val.shape}, Validation target shape: {y_val.shape}")
        
        # Define XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'scale_pos_weight':50,
            'random_state': 42
        }
        
        print("Creating XGBoost classifier...")
        # Create XGBoost classifier
        model = xgb.XGBClassifier(**params)
        
        # Train the model
        print("Training XGBoost model...")
        model.fit(X_train, y_train)
        print("Model training completed successfully")
        
        # Create output directory if it doesn't exist
        print(f"Output directory: {model_output.path}")
        os.makedirs(model_output.path, exist_ok=True)
        print(f"Output directory confirmed to exist")
        
        # Save the trained model with the correct filename for Vertex AI
        model_file = os.path.join(model_output.path, 'model.pkl')
        print(f"Saving model to: {model_file}")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved successfully")
        
        # Make predictions on validation set
        print("Making predictions on validation set...")
        y_pred = model.predict(X_val)
        
        # Calculate confusion matrix
        confusion = confusion_matrix(y_val, y_pred)
        
        # Create and save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_file = os.path.join(model_output.path, 'confusion_matrix.png')
        print(f"Saving confusion matrix to: {cm_file}")
        plt.savefig(cm_file)
        print(f"Confusion matrix saved successfully")
        
        print("Model training and evaluation completed")
        print("Confusion Matrix:")
        print(confusion)
        
    except Exception as e:
        print(f"Error in train_xgboost_model: {str(e)}")
        print(traceback.format_exc())
        raise e

# Enhanced Confusion Matrix Component
@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas", 
        "scikit-learn", 
        "matplotlib", 
        "numpy", 
        "seaborn", 
        "fsspec", 
        "gcsfs",
        "xgboost"
    ],
)
def evaluate_model_metrics(
    test_data: Input[Dataset],
    model_path: Input[Model],
    metrics_output: Output[Metrics],
    confusion_matrix_artifact: Output[Dataset]
):
    import pandas as pd
    import numpy as np
    import pickle
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import json
    import traceback
    
    try:
        print(f"Loading test data from: {test_data.path}")
        # Load test data
        test_df = pd.read_csv(test_data.path)
        print(f"Test data loaded successfully. Shape: {test_df.shape}")
        
        # Split features and target
        X_test = test_df.drop(['Class'], axis=1)
        y_test = test_df['Class']
        
        # Load the trained model
        model_file = os.path.join(model_path.path, 'model.pkl')
        print(f"Loading model from: {model_file}")
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully")
        
        # Make predictions
        print(f"Making predictions on test data...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probability for positive class
        
        # Calculate metrics
        print("Calculating performance metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Create detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Convert confusion matrix to a more detailed format
        tn, fp, fn, tp = cm.ravel()
        cm_detailed = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        # Save metrics to a file
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm_detailed,
            'classification_report': report
        }
        
        # Create a more detailed confusion matrix visualization
        plt.figure(figsize=(10, 8))
        
        # Main confusion matrix
        ax1 = plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_title('Confusion Matrix')
        
        # Normalized confusion matrix
        ax2 = plt.subplot(2, 2, 2)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_title('Normalized Confusion Matrix')
        
        # Create bar chart for precision, recall, and f1
        ax3 = plt.subplot(2, 2, 3)
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
            'Value': [precision, recall, f1, accuracy]
        })
        sns.barplot(x='Metric', y='Value', data=metrics_df, ax=ax3)
        ax3.set_ylim(0, 1)
        ax3.set_title('Model Performance Metrics')
        
        # Add text with detailed metrics
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        details = (
            f"Model Performance Details:\n\n"
            f"Total Samples: {len(y_test)}\n"
            f"Positive Samples (Fraud): {sum(y_test)}\n"
            f"Negative Samples (Normal): {len(y_test) - sum(y_test)}\n\n"
            f"True Positives: {tp}\n"
            f"False Positives: {fp}\n"
            f"True Negatives: {tn}\n"
            f"False Negatives: {fn}\n\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"F1 Score: {f1:.4f}\n"
            f"AUC: {auc:.4f}"
        )
        ax4.text(0, 1, details, fontsize=10, va='top')
        
        # Adjust layout and save figure
        plt.tight_layout()
        
        # Ensure output directories exist
        os.makedirs(os.path.dirname(confusion_matrix_artifact.path), exist_ok=True)
        
        # Save the visualization
        cm_figure_path = confusion_matrix_artifact.path
        plt.savefig(cm_figure_path, dpi=300, bbox_inches='tight')
        
        # Also save data as CSV for easier access
        cm_df = pd.DataFrame(cm, 
                      index=['Actual Negative', 'Actual Positive'], 
                      columns=['Predicted Negative', 'Predicted Positive'])
        csv_path = os.path.join(os.path.dirname(confusion_matrix_artifact.path), 'confusion_matrix.csv')
        cm_df.to_csv(csv_path)
        
        # Save metrics as JSON
        with open(metrics_output.path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("Model evaluation completed successfully")
        print(f"Metrics saved to: {metrics_output.path}")
        print(f"Confusion matrix visualization saved to: {cm_figure_path}")
        
    except Exception as e:
        print(f"Error in evaluate_model_metrics: {str(e)}")
        print(traceback.format_exc())
        raise e

# Simple prediction component for immediate predictions
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "xgboost", "fsspec", "gcsfs"],
)
def predict_with_model(
    test_data: Input[Dataset],
    model_path: Input[Model],
    predictions_output: Output[Dataset]
):
    import pandas as pd
    import numpy as np
    import pickle
    import os
    import traceback
    
    try:
        print(f"Loading test data from: {test_data.path}")
        # Load test data
        test_df = pd.read_csv(test_data.path)
        print(f"Test data loaded successfully. Shape: {test_df.shape}")
        
        # Split features and target
        X_test = test_df.drop(['Class'], axis=1)
        y_test = test_df['Class']
        
        # Check if model file exists
        model_file = os.path.join(model_path.path, 'model.pkl')
        if not os.path.exists(model_file):
            print(f"Model file not found at: {model_file}")
            # List files in model directory
            if os.path.exists(model_path.path):
                print(f"Files in model directory:")
                for file in os.listdir(model_path.path):
                    print(f"  - {file}")
            else:
                print(f"Model directory does not exist: {model_path.path}")
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        print(f"Loading model from: {model_file}")
        # Load the trained model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(predictions_output.path), exist_ok=True)
        
        # Make predictions
        print(f"Making predictions on test data...")
        y_pred = model.predict(X_test)
        print(f"Predictions complete")
        
        # Create a dataframe with the predictions
        result_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred
        })
        
        # Save predictions
        print(f"Saving predictions to: {predictions_output.path}")
        result_df.to_csv(predictions_output.path, index=False)
        print("Predictions saved successfully")
        
    except Exception as e:
        print(f"Error in predict_with_model: {str(e)}")
        print(traceback.format_exc())
        raise e

# Batch prediction component (replaces deployment component)
@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform", "fsspec", "gcsfs"],
)
def create_batch_prediction_job(
    model_path: Input[Model],
    test_data: Input[Dataset],
    project_id: str,
    region: str,
    model_display_name: str,
    batch_output_path: str,
    machine_type: str = "n1-standard-2"
) -> str:
    from google.cloud import aiplatform
    import os
    import traceback
    
    try:
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        
        # Convert paths if needed
        gcs_model_path = model_path.path
        if gcs_model_path.startswith('/gcs/'):
            bucket_name = gcs_model_path.split('/')[2]
            path_suffix = '/'.join(gcs_model_path.split('/')[3:])
            gcs_model_path = f"gs://{bucket_name}/{path_suffix}"
        
        test_data_path = test_data.path
        if test_data_path.startswith('/gcs/'):
            bucket_name = test_data_path.split('/')[2]
            path_suffix = '/'.join(test_data_path.split('/')[3:])
            test_data_path = f"gs://{bucket_name}/{path_suffix}"
        
        print(f"Uploading model from {gcs_model_path} to Vertex AI")
        
        # Verify model file exists with the correct name
        model_file = os.path.join(model_path.path, 'model.pkl')
        if os.path.exists(model_file):
            print(f"Verified that model.pkl exists at {model_file}")
        else:
            print(f"Warning: model.pkl not found at {model_file}")
            print("Listing directory contents:")
            if os.path.exists(model_path.path):
                for f in os.listdir(model_path.path):
                    print(f"  - {f}")
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Upload model to Vertex AI
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=gcs_model_path,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-4:latest",
            sync=True
        )
        
        print(f"Model uploaded successfully with ID: {model.resource_name}")
        
        # Create batch prediction job
        print(f"Creating batch prediction job with test data: {test_data_path}")
        batch_prediction_job = model.batch_predict(
            job_display_name=f"{model_display_name}-batch-job",
            gcs_source=test_data_path,
            gcs_destination_prefix=batch_output_path,
            machine_type=machine_type,
            sync=True
        )
        
        print(f"Batch prediction job created: {batch_prediction_job.resource_name}")
        print(f"Batch prediction results will be available at: {batch_output_path}")
        
        # Return batch prediction job resource name
        return batch_prediction_job.resource_name
        
    except Exception as e:
        print(f"Error in batch prediction: {e}")
        print(traceback.format_exc())
        # Return error message instead of raising exception
        return f"Batch prediction failed: {str(e)}"

# Define the pipeline
@dsl.pipeline(
    name="credit-card-fraud-ml-pipeline",
    description="Pipeline for credit card fraud detection with preprocessing, training, and batch prediction with enhanced metrics"
)
def credit_card_fraud_pipeline(
    data_path: str,
    project_id: str = PROJECT_ID,
    region: str = REGION,
    run_batch_prediction: bool = True,
    export_metrics: bool = True,
    model_display_name: str = "credit-card-fraud-xgboost",
    machine_type: str = "n1-standard-2",
    batch_output_path: str = f"gs://{BUCKET_NAME}/batch_predictions/"
):
    # Preprocess the data
    preprocess_task = preprocess_data(data_path=data_path)
    
    # Split the preprocessed data
    split_task = split_data(
        processed_data_path=preprocess_task.outputs["processed_data_output"]
    )
    
    # Train the XGBoost model
    train_task = train_xgboost_model(
        train_data=split_task.outputs["train_data"],
        validation_data=split_task.outputs["validation_data"]
    )
    
    # Make immediate predictions with the component
    predict_task = predict_with_model(
        test_data=split_task.outputs["test_data"],
        model_path=train_task.outputs["model_output"]
    )
    
    # Add enhanced model evaluation with detailed confusion matrix
    if export_metrics:
        evaluate_task = evaluate_model_metrics(
            test_data=split_task.outputs["test_data"],
            model_path=train_task.outputs["model_output"]
        )
    
    # Run batch prediction job if requested
    if run_batch_prediction:
        batch_predict_task = create_batch_prediction_job(
            model_path=train_task.outputs["model_output"],
            test_data=split_task.outputs["test_data"],
            project_id=project_id,
            region=region,
            model_display_name=model_display_name,
            batch_output_path=batch_output_path,
            machine_type=machine_type
        )

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection ML Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Compile the pipeline")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the pipeline")
    run_parser.add_argument("--data_path", required=True, help="Path to input data")
    run_parser.add_argument("--project_id", default=PROJECT_ID, help="GCP Project ID")
    run_parser.add_argument("--region", default=REGION, help="GCP Region")
    run_parser.add_argument("--pipeline_root", default=PIPELINE_ROOT, help="GCS Pipeline Root")
    run_parser.add_argument("--run_batch_prediction", type=bool, default=True, help="Whether to run batch prediction")
    run_parser.add_argument("--export_metrics", type=bool, default=True, help="Whether to export enhanced metrics with confusion matrix")
    run_parser.add_argument("--model_display_name", default=None, help="Model display name")
    run_parser.add_argument("--machine_type", default="n1-standard-2", help="Machine type for model deployment")
    run_parser.add_argument("--batch_output_path", default=f"gs://{BUCKET_NAME}/batch_predictions/", help="Path for batch prediction outputs")
    
    return parser.parse_args()

# Compile the pipeline
def compile_pipeline():
    compiler.Compiler().compile(
        pipeline_func=credit_card_fraud_pipeline,
        package_path="credit_card_fraud_ml_pipeline.json"
    )
    print("Pipeline compiled successfully to credit_card_fraud_ml_pipeline.json")

# Run the pipeline on Vertex AI
def run_pipeline(args):
    # Initialize Vertex AI
    aiplatform.init(project=args.project_id, location=args.region)
    
    # Set model display name with timestamp if not provided
    model_display_name = args.model_display_name
    if model_display_name is None:
        model_display_name = "credit-card-fraud-xgboost-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create pipeline job
    job = aiplatform.PipelineJob(
        display_name="credit-card-fraud-ml",
        template_path="credit_card_fraud_ml_pipeline.json",
        pipeline_root=args.pipeline_root,
        parameter_values={
            "data_path": args.data_path,
            "project_id": args.project_id,
            "region": args.region,
            "run_batch_prediction": args.run_batch_prediction,
            "export_metrics": args.export_metrics,
            "model_display_name": model_display_name,
            "machine_type": args.machine_type,
            "batch_output_path": args.batch_output_path
        },
        enable_caching=True
    )
    
    # Run the pipeline
    job.run()
    print(f"Pipeline job submitted with ID: {job.name}")
    return job

if __name__ == "__main__":
    args = parse_args()
    
    if args.command == "compile":
        compile_pipeline()
    elif args.command == "run":
        run_pipeline(args)
    else:
        print("Please specify a command: compile or run")