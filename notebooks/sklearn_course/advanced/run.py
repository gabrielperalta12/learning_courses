"""
run.py — Orquestador del pipeline completo: train → evaluate → predict.

Uso:
  uv run python run.py                        # pipeline completo con defaults
  uv run python run.py --trials 100           # más trials de Optuna
  uv run python run.py --no-optuna            # skip búsqueda, hiperparámetros default
  uv run python run.py --skip-train           # solo evaluate + predict (modelo ya existe)
  uv run python run.py --input nuevos.csv     # predict sobre archivo externo
"""
import argparse
import time
from pathlib import Path

from config import MODEL_PATH, NUM_FEATURES, CAT_FEATURES, N_TRIALS


def run_train(n_trials: int, use_optuna: bool) -> dict:
    from train import train
    print("\n" + "=" * 50)
    print("PASO 1 — Entrenamiento")
    print("=" * 50)
    return train(n_trials=n_trials, use_optuna=use_optuna)


def run_evaluate() -> dict:
    from evaluate import evaluate
    print("\n" + "=" * 50)
    print("PASO 2 — Evaluación")
    print("=" * 50)
    return evaluate()


def run_predict(input_path: Path, output_path: Path):
    import pandas as pd
    from predict import load_artifact, predict_dataframe

    print("\n" + "=" * 50)
    print("PASO 3 — Predicción")
    print("=" * 50)

    artifact = load_artifact(MODEL_PATH)
    df = pd.read_csv(input_path)
    predictions = predict_dataframe(df, artifact, include_features=True)
    predictions.to_csv(output_path, index=False)
    print(f"→ {len(predictions)} predicciones guardadas en: {output_path}")
    print(f"   Score medio:     {predictions['conversion_score'].mean():.4f}")
    print(f"   Tasa conversión: {predictions['conversion_label'].mean():.2%}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo de conversión")
    parser.add_argument("--trials",      type=int,  default=N_TRIALS,       help="Trials de Optuna")
    parser.add_argument("--no-optuna",   action="store_true",                help="Hiperparámetros default")
    parser.add_argument("--skip-train",  action="store_true",                help="Saltar entrenamiento")
    parser.add_argument("--input",       type=Path, default=None,            help="CSV para predicción")
    parser.add_argument("--output",      type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    t_start = time.time()

    if not args.skip_train:
        run_train(n_trials=args.trials, use_optuna=not args.no_optuna)
    else:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"--skip-train requiere un modelo en {MODEL_PATH}. Ejecutá sin el flag primero.")
        print(f"\n→ Saltando entrenamiento — usando modelo existente: {MODEL_PATH}")

    run_evaluate()

    if args.input:
        run_predict(args.input, args.output)
    else:
        print("\n→ Sin --input: paso de predicción saltado.")
        print(f"   Para predecir: uv run python run.py --skip-train --input datos.csv")

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Pipeline completado en {elapsed:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
