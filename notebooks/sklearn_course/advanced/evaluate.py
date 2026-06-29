"""
evaluate.py — Métricas, curvas y reporte de un modelo guardado.

Carga el artifact de train.py y genera:
  - Tabla de métricas (AUC, Acc, F1, Precision, Recall)
  - Curva ROC
  - Curva Precision-Recall
  - Feature importances (Gini + Permutation)
  - Calibration curve

Uso:
  python evaluate.py                    # usa MODEL_PATH de config
  python evaluate.py --model path/a/modelo.pkl
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

from config import (
    CAT_FEATURES,
    MODEL_PATH,
    NUM_FEATURES,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_CLF,
    TEST_SIZE,
)
from train import generate_dataset

plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e8e8", "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def load_artifact(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def print_metrics(y_true, y_pred, y_prob, label: str = "Test") -> dict:
    metrics = {
        "auc":       round(roc_auc_score(y_true, y_prob), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
    }
    print(f"\n── {label} Metrics ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    return metrics


def plot_roc_pr(y_true, y_prob, save_path: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, color="#4361ee", linewidth=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#aaa", label="Azar")
    axes[0].set_xlabel("FPR (1 - Specificity)")
    axes[0].set_ylabel("TPR (Recall)")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    no_skill = y_true.mean()
    axes[1].plot(rec, prec, color="#f72585", linewidth=2)
    axes[1].axhline(no_skill, color="#aaa", linestyle="--", label=f"No skill ({no_skill:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Gráfica guardada: {save_path}")
    plt.show()


def plot_feature_importance(
    model, X_test_enc: np.ndarray, y_test, feat_names: list[str],
    top_n: int = 12, save_path: Path | None = None,
):
    gini_imp = pd.Series(model.feature_importances_, index=feat_names)

    perm = permutation_importance(
        model, X_test_enc, y_test,
        n_repeats=15, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    perm_imp = pd.Series(perm.importances_mean, index=feat_names)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, imp, title, color in [
        (axes[0], gini_imp.sort_values(ascending=False).head(top_n), "Gini Importance", "#4361ee"),
        (axes[1], perm_imp.sort_values(ascending=False).head(top_n), "Permutation Importance (AUC)", "#f72585"),
    ]:
        ax.barh(imp.index[::-1], imp.values[::-1], color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Importancia")

    plt.suptitle("Feature Importances", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Gráfica guardada: {save_path}")
    plt.show()


def plot_calibration(y_true, y_prob, save_path: Path | None = None):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Calibración perfecta")
    ax.plot(mean_pred, frac_pos, "o-", color="#4361ee", linewidth=2, label="Modelo")
    ax.set_xlabel("Probabilidad media predicha")
    ax.set_ylabel("Fracción de positivos reales")
    ax.set_title("Calibration Curve")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def evaluate(model_path: Path = MODEL_PATH):
    print(f"→ Cargando modelo: {model_path}")
    artifact     = load_artifact(model_path)
    preprocessor = artifact["preprocessor"]
    model        = artifact["model"]
    feat_names   = artifact["feature_names"]

    df = generate_dataset()
    X  = df[NUM_FEATURES + CAT_FEATURES]
    y  = df[TARGET_CLF]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_test_enc = preprocessor.transform(X_test)
    y_prob     = model.predict_proba(X_test_enc)[:, 1]
    y_pred     = model.predict(X_test_enc)

    metrics = print_metrics(y_test, y_pred, y_prob)

    plot_roc_pr(y_test, y_prob,     save_path=REPORTS_DIR / "roc_pr.png")
    plot_feature_importance(model, X_test_enc, y_test, feat_names,
                            save_path=REPORTS_DIR / "feature_importance.png")
    plot_calibration(y_test, y_prob, save_path=REPORTS_DIR / "calibration.png")

    # Actualizar reporte con métricas de evaluación
    report_path = REPORTS_DIR / "evaluation_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
    else:
        report = {}

    report["evaluation_metrics"] = metrics
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n→ Reporte actualizado: {report_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar modelo de conversión")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    evaluate(model_path=args.model)
