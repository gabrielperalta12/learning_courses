"""
predict.py — Inferencia sobre datos nuevos usando el modelo entrenado.

Carga el artifact (preprocessor + model) y expone:
  - predict_dataframe(df)  → DataFrame con scores y labels
  - predict_record(record) → dict con score y label para un registro
  - CLI: python predict.py --input data.csv --output predictions.csv

El preprocesador ya fue fiteado en train.py → no hay data leakage.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import CAT_FEATURES, MODEL_PATH, NUM_FEATURES, TARGET_CLF


def load_artifact(path: Path = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_dataframe(
    df: pd.DataFrame,
    artifact: dict,
    threshold: float = 0.5,
    include_features: bool = False,
) -> pd.DataFrame:
    """
    Aplica el pipeline a un DataFrame y devuelve scores + labels.

    Args:
        df:               DataFrame con las columnas de features (sin target).
        artifact:         Dict con 'preprocessor' y 'model' (salida de train.py).
        threshold:        Umbral de clasificación (default 0.5).
        include_features: Si True, incluye las features originales en el output.

    Returns:
        DataFrame con columnas: conversion_score, conversion_label (+ features si se pide).
    """
    preprocessor = artifact["preprocessor"]
    model        = artifact["model"]
    feat_names   = artifact["feature_names"]

    missing = [c for c in feat_names if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en el input: {missing}")

    X_enc  = preprocessor.transform(df[feat_names])
    scores = model.predict_proba(X_enc)[:, 1]
    labels = (scores >= threshold).astype(int)

    result = pd.DataFrame({
        "conversion_score": scores.round(4),
        "conversion_label": labels,
    }, index=df.index)

    if include_features:
        result = pd.concat([df[feat_names], result], axis=1)

    return result


def predict_record(record: dict, artifact: dict, threshold: float = 0.5) -> dict:
    """Predice para un único registro (dict con los valores de features)."""
    df  = pd.DataFrame([record])
    out = predict_dataframe(df, artifact, threshold=threshold)
    return {
        "conversion_score": float(out["conversion_score"].iloc[0]),
        "conversion_label": int(out["conversion_label"].iloc[0]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia de conversión")
    parser.add_argument("--input",     type=Path, required=True,    help="CSV con features")
    parser.add_argument("--output",    type=Path, required=True,    help="CSV de salida con scores")
    parser.add_argument("--model",     type=Path, default=MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.5,    help="Umbral de clasificación")
    args = parser.parse_args()

    print(f"→ Cargando modelo: {args.model}")
    artifact = load_artifact(args.model)

    print(f"→ Leyendo input: {args.input}")
    df_input = pd.read_csv(args.input)

    print(f"→ Generando predicciones para {len(df_input)} registros...")
    predictions = predict_dataframe(
        df_input, artifact,
        threshold=args.threshold,
        include_features=True,
    )

    predictions.to_csv(args.output, index=False)
    print(f"→ Predicciones guardadas: {args.output}")

    positive_rate = predictions["conversion_label"].mean()
    print(f"   Score medio:     {predictions['conversion_score'].mean():.4f}")
    print(f"   Tasa conversión: {positive_rate:.2%} (umbral={args.threshold})")
