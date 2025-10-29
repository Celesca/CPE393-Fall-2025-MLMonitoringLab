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

def build_pipeline():
    # TODO: you may try different models or parameters
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

    # ---- MLflow run ----
    with mlflow.start_run(run_name="baseline_LR"):
        model = build_pipeline()

        # TODO: log your chosen hyperparameters explicitly
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("penalty", "l2")
        mlflow.log_param("C", 1.0)

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
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["0","1"])
        plt.yticks(tick_marks, ["0","1"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        fig_path = REPORTS_DIR / "confusion_matrix.png"
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
            plt.title("ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            roc_path = REPORTS_DIR / "roc_curve.png"
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
            artifact_path="model",       # MLflow still accepts artifact_path here
            input_example=input_example, # <-- ADDED
            signature=signature          # <-- ADDED
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
            ClassificationPreset()  # 0.4.33 name
        ])
        report.run(reference_data=ref_df_copy, current_data=cur_df_copy, column_mapping=column_mapping)

        html_path = REPORTS_DIR / "evidently_report.html"
        json_path = REPORTS_DIR / "evidently_report.json"
        report.save_html(str(html_path))
        report.save_json(str(json_path))

        # Log Evidently artifacts to MLflow
        mlflow.log_artifact(str(html_path))
        mlflow.log_artifact(str(json_path))

        print("Run complete. Check MLflow UI and the reports/ folder.")

if __name__ == "__main__":
    ref, cur = load_data()
    train_and_log(ref, cur)
