import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset
from evidently import ColumnMapping

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
REPORTS_DIR = BASE / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

TRACKING_URI = "http://127.0.0.1:8080"  # set your tracking URI here
mlflow.set_tracking_uri(TRACKING_URI)
EXP_NAME = "mlflow-evidently-lab"
mlflow.set_experiment(EXP_NAME)

def load_data():
    ref = pd.read_csv(DATA_DIR / "train.csv")
    cur = pd.read_csv(DATA_DIR / "test.csv")
    return ref, cur

def build_pipeline(model_type="lr"):
    """
    Build a pipeline with a specified model.
    
    Args:
        model_type (str): "lr" for LogisticRegression, "rf" for RandomForest
    
    Returns:
        Pipeline: sklearn Pipeline with scaler and classifier
    """
    if model_type == "rf":
        # RandomForest doesn't require scaling, but we include it for consistency
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
    else:  # default to LogisticRegression
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=None))
        ])
    return model

def train_and_log(ref_df, cur_df):
    target = "target"
    X_train = ref_df.drop(columns=[target])
    y_train = ref_df[target].astype(int)

    X_test = cur_df.drop(columns=[target])
    y_test = cur_df[target].astype(int)

    # Train and log both models
    models_config = [
        {
            "name": "baseline_LR",
            "model_type": "lr",
            "params": {
                "model": "LogisticRegression",
                "scaler": "StandardScaler",
                "penalty": "l2",
                "C": 1.0,
                "max_iter": 1000
            }
        },
        {
            "name": "baseline_RF",
            "model_type": "rf",
            "params": {
                "model": "RandomForestClassifier",
                "scaler": "StandardScaler",
                "n_estimators": 100,
                "random_state": 42
            }
        }
    ]
    
    for config in models_config:
        _train_single_model(
            X_train, y_train, X_test, y_test, 
            ref_df, cur_df, target,
            config["name"], config["model_type"], config["params"]
        )


def _train_single_model(X_train, y_train, X_test, y_test, ref_df, cur_df, target, run_name, model_type, params):
    """
    Train and log a single model to MLflow.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        ref_df, cur_df: Reference and current dataframes for Evidently
        target: Target column name
        run_name: Name of the MLflow run
        model_type: Type of model ("lr" or "rf")
        params: Dictionary of parameters to log
    """
    # ---- MLflow run ----
    with mlflow.start_run(run_name=run_name):
        model = build_pipeline(model_type=model_type)

        # Log hyperparameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        model.fit(X_train, y_train)

        # Predictions & metrics
        y_pred = model.predict(X_test)
        if hasattr(model.named_steps["clf"], "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            mlflow.log_metric("roc_auc", float(roc_auc))
        else:
            y_proba = None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1", float(f1))

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix - {run_name}")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["0","1"])
        plt.yticks(tick_marks, ["0","1"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        fig_path = REPORTS_DIR / f"confusion_matrix_{run_name}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(fig_path))

        # ROC curve (if proba available)
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig2 = plt.figure()
            plt.plot(fpr, tpr, label="ROC")
            plt.plot([0,1],[0,1], linestyle="--")
            plt.title(f"ROC Curve - {run_name}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            roc_path = REPORTS_DIR / f"roc_curve_{run_name}.png"
            plt.tight_layout()
            plt.savefig(roc_path, dpi=150)
            plt.close(fig2)
            mlflow.log_artifact(str(roc_path))

        # ---- Log the sklearn pipeline as a model (with signature + input_example) ----
        input_example = X_train.head(5)
        try:
            signature = infer_signature(X_train, model.predict(X_train))
        except Exception:
            signature = None

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        # ---- Evidently report ----
        ref_df_copy = ref_df.copy()
        cur_df_copy = cur_df.copy()

        # Add predictions & proba columns for Evidently mapping (CURRENT set)
        cur_df_copy["prediction"] = y_pred.astype(int)
        if y_proba is not None:
            cur_df_copy["proba_1"] = y_proba
            cur_df_copy["proba_0"] = 1.0 - y_proba

        # For REFERENCE set, compute predictions/probas using X_train to avoid extra columns issues
        y_ref_pred = model.predict(X_train)
        ref_df_copy["prediction"] = y_ref_pred.astype(int)
        if hasattr(model.named_steps["clf"], "predict_proba"):
            ref_proba = model.predict_proba(X_train)[:, 1]
            ref_df_copy["proba_1"] = ref_proba
            ref_df_copy["proba_0"] = 1.0 - ref_proba

        # Build ColumnMapping via attributes (required in 0.4.33)
        column_mapping = ColumnMapping()
        column_mapping.target = target
        column_mapping.prediction = "prediction"
        column_mapping.prediction_probas = ["proba_0", "proba_1"]

        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            ClassificationPreset()
        ])
        report.run(reference_data=ref_df_copy, current_data=cur_df_copy, column_mapping=column_mapping)

        html_path = REPORTS_DIR / f"evidently_report_{run_name}.html"
        json_path = REPORTS_DIR / f"evidently_report_{run_name}.json"
        report.save_html(str(html_path))
        report.save_json(str(json_path))

        # Log Evidently artifacts to MLflow
        mlflow.log_artifact(str(html_path))
        mlflow.log_artifact(str(json_path))

        # ---- Log additional metrics for monitoring ----
        # Calculate and log class distribution metrics
        current_pos_ratio = (y_test == 1).sum() / len(y_test)
        reference_pos_ratio = (y_train == 1).sum() / len(y_train)
        
        mlflow.log_metric("reference_positive_ratio", float(reference_pos_ratio))
        mlflow.log_metric("current_positive_ratio", float(current_pos_ratio))
        mlflow.log_metric("class_distribution_drift", float(abs(current_pos_ratio - reference_pos_ratio)))
        
        # Calculate recall and precision for monitoring
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mlflow.log_metric("precision", float(precision))
        mlflow.log_metric("recall", float(recall))
        
        print(f"Run '{run_name}' complete.")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Class Distribution Drift: {abs(current_pos_ratio - reference_pos_ratio):.4f}")
        print(f"Check MLflow UI and the reports/ folder.")

if __name__ == "__main__":
    ref, cur = load_data()
    train_and_log(ref, cur)
