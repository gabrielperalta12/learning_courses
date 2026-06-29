"""
train.py — Entrenamiento con Optuna + XGBoost, guarda el modelo final.

Flujo:
  1. Carga / genera datos
  2. Split train / test
  3. Optuna busca hiperparámetros óptimos (hold-out interno para early stopping)
  4. Reentrena modelo final con todos los datos de train
  5. Persiste modelo + metadatos con joblib

Uso:
  python train.py
  python train.py --trials 100 --no-optuna   # usa hiperparámetros default
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from config import (
    CAT_FEATURES,
    CV_FOLDS,
    EARLY_STOPPING_RDS,
    MODEL_PATH,
    N_TRIALS,
    NUM_FEATURES,
    OPTUNA_DB,
    RANDOM_STATE,
    REPORT_PATH,
    SCORING,
    STUDY_NAME,
    TARGET_CLF,
    TEST_SIZE,
    VAL_SIZE,
    XGB_PARAM_SPACE,
)
from features import build_preprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Datos de ejemplo (reemplazar con carga real: pd.read_csv / query a BD)
# ---------------------------------------------------------------------------
def generate_dataset(n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    sessions       = rng.integers(1, 30, n)
    time_on_site   = rng.uniform(30, 900, n)
    pages          = rng.uniform(1, 12, n)
    days_since_reg = rng.integers(0, 60, n)
    features_used  = rng.integers(0, 15, n)
    errors_seen    = rng.poisson(1.5, n)
    channel        = rng.choice(["organic", "paid", "email", "direct", "referral"], n)
    device         = rng.choice(["mobile", "desktop", "tablet"], n)
    plan           = rng.choice(["free", "trial", "pro", "enterprise"], n, p=[0.55, 0.25, 0.15, 0.05])
    country        = rng.choice(["US", "UK", "DE", "MX", "BR", "other"], n)

    logit = (
        -4 + sessions * 0.10 + time_on_site * 0.0018 + pages * 0.15
        - days_since_reg * 0.05 + features_used * 0.22 - errors_seen * 0.3
        + np.where(channel == "email", 1.0, 0)
        + np.where(device == "desktop", 0.5, 0)
        + np.where(plan == "trial", 1.3, np.where(plan == "pro", 2.1, np.where(plan == "enterprise", 3.0, 0)))
    )
    prob      = 1 / (1 + np.exp(-logit))
    converted = (rng.uniform(0, 1, n) < prob).astype(int)

    return pd.DataFrame({
        "sessions": sessions, "time_on_site": time_on_site.round(0),
        "pages": pages.round(2), "days_since_reg": days_since_reg,
        "features_used": features_used, "errors_seen": errors_seen,
        "channel": channel, "device": device, "plan": plan, "country": country,
        "converted": converted,
    })


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def _suggest_params(trial: optuna.Trial) -> dict:
    """Traduce XGB_PARAM_SPACE a llamadas trial.suggest_*."""
    params = {}
    for name, spec in XGB_PARAM_SPACE.items():
        kind, *bounds = spec
        if kind == "float_log":
            params[name] = trial.suggest_float(name, *bounds, log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, *bounds)
        elif kind == "int":
            params[name] = trial.suggest_int(name, *bounds)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, bounds)
    return params


def build_objective(X_tr, y_tr, X_val, y_val):
    def objective(trial: optuna.Trial) -> float:
        params = {
            **_suggest_params(trial),
            "n_estimators":          1000,
            "early_stopping_rounds": EARLY_STOPPING_RDS,
            "eval_metric":           "auc",
            "random_state":          RANDOM_STATE,
            "verbosity":             0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return model.best_score

    return objective


# ---------------------------------------------------------------------------
# Entrenamiento principal
# ---------------------------------------------------------------------------
def train(n_trials: int = N_TRIALS, use_optuna: bool = True) -> dict:
    print("→ Cargando datos...")
    df = generate_dataset()

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET_CLF]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_enc  = preprocessor.fit_transform(X_train)
    X_test_enc   = preprocessor.transform(X_test)

    # Hold-out interno para early stopping durante Optuna
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_enc, y_train, test_size=VAL_SIZE, random_state=0
    )

    # ------------------------------------------------------------------
    # Búsqueda de hiperparámetros
    # ------------------------------------------------------------------
    if use_optuna:
        print(f"→ Optuna: {n_trials} trials (TPE)...")
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=OPTUNA_DB,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            load_if_exists=True,
        )
        study.optimize(
            build_objective(X_tr, y_tr, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        best_params = study.best_params
        print(f"   Mejor Val AUC: {study.best_value:.4f}")
    else:
        # Hiperparámetros default razonables sin búsqueda
        best_params = {
            "learning_rate": 0.05, "max_depth": 6,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "reg_alpha": 0.1, "reg_lambda": 1.0, "gamma": 0.0,
        }

    # ------------------------------------------------------------------
    # Modelo final — reentrenar con todos los datos de train
    # ------------------------------------------------------------------
    print("→ Entrenando modelo final...")
    final_model = xgb.XGBClassifier(
        **best_params,
        n_estimators=1000,
        early_stopping_rounds=EARLY_STOPPING_RDS,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # Validación cruzada sobre todo el train (para estimado de generalización)
    cv_scores = cross_val_score(
        xgb.XGBClassifier(**best_params, n_estimators=final_model.best_iteration,
                           random_state=RANDOM_STATE, verbosity=0, eval_metric="logloss"),
        X_train_enc, y_train,
        cv=CV_FOLDS, scoring=SCORING, n_jobs=-1,
    )

    test_auc = roc_auc_score(y_test, final_model.predict_proba(X_test_enc)[:, 1])

    print(f"   CV AUC:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Test AUC: {test_auc:.4f}")

    # ------------------------------------------------------------------
    # Persistir modelo + preprocesador
    # ------------------------------------------------------------------
    artifact = {
        "preprocessor": preprocessor,
        "model":        final_model,
        "best_params":  best_params,
        "feature_names": NUM_FEATURES + CAT_FEATURES,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    metadata = {
        "cv_auc_mean":  round(float(cv_scores.mean()), 4),
        "cv_auc_std":   round(float(cv_scores.std()), 4),
        "test_auc":     round(float(test_auc), 4),
        "best_params":  best_params,
        "n_estimators": int(final_model.best_iteration),
        "train_rows":   len(X_train),
        "test_rows":    len(X_test),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"   Modelo guardado en: {MODEL_PATH}")
    print(f"   Reporte guardado en: {REPORT_PATH}")
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo de conversión")
    parser.add_argument("--trials",     type=int,  default=N_TRIALS, help="Número de trials Optuna")
    parser.add_argument("--no-optuna",  action="store_true",          help="Usar hiperparámetros default")
    args = parser.parse_args()

    result = train(n_trials=args.trials, use_optuna=not args.no_optuna)
    print("\nMetadatos finales:")
    for k, v in result.items():
        if k != "best_params":
            print(f"  {k}: {v}")
