# Advanced — ML Pipelines en Python

Etapa avanzada del curso: los conceptos explorados en los notebooks se traducen a **código de producción** organizado en módulos.

## Estructura

```
advanced/
├── config.py      # Punto único de configuración: rutas, features, param spaces
├── features.py    # Construcción del preprocesador sklearn (ColumnTransformer)
├── train.py       # Entrenamiento con Optuna + XGBoost, persiste el modelo
├── evaluate.py    # Métricas, curvas ROC/PR, feature importances, calibración
├── predict.py     # Inferencia sobre datos nuevos
└── run.py         # Orquestador: ejecuta train → evaluate → predict en secuencia
```

## Setup

```bash
# Instalar dependencias (uv resuelve y sincroniza el entorno desde pyproject.toml)
uv sync
```

## Flujo completo

```bash
# Pipeline completo de una vez (train → evaluate → predict)
uv run python run.py
uv run python run.py --trials 100                              # más trials de Optuna
uv run python run.py --no-optuna                               # hiperparámetros default
uv run python run.py --input nuevos.csv --output scores.csv   # incluye predicción
uv run python run.py --skip-train --input nuevos.csv          # solo evaluate + predict

# O paso a paso
uv run python train.py
uv run python train.py --no-optuna
uv run python evaluate.py
uv run python predict.py --input data/nuevos_usuarios.csv --output data/scores.csv
```

## Qué enseña cada módulo

| Módulo | Concepto clave |
|---|---|
| `config.py` | Separar configuración de lógica — un solo lugar para cambiar rutas, features o hiperparámetros |
| `features.py` | `ColumnTransformer` + `Pipeline` como objetos reutilizables — se fitean una vez, se reúsan en train/predict |
| `train.py` | Optuna con early stopping, persistencia con `pickle`, CLI con `argparse` |
| `evaluate.py` | Evaluación reproducible y guardado de reportes — separado del entrenamiento |
| `predict.py` | Inferencia sin reentrenar el preprocesador — cero data leakage |

## Notebooks relacionados

- `08_pipelines.ipynb` — fundamentos de sklearn Pipeline
- `09b_gradient_boosting_frameworks.ipynb` — XGBoost, LightGBM, CatBoost
- `09c_optuna_hyperparameter_tuning.ipynb` — Optuna en profundidad
