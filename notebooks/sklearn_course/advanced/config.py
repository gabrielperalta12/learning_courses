"""
config.py — Constantes, rutas y espacios de hiperparámetros.
Punto único de configuración: cambiar aquí afecta todo el pipeline.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[4]   # raíz del repo
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH  = MODELS_DIR / "conversion_model.pkl"
REPORT_PATH = REPORTS_DIR / "evaluation_report.json"

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
NUM_FEATURES = [
    "sessions",
    "time_on_site",
    "pages",
    "days_since_reg",
    "features_used",
    "errors_seen",
]

CAT_FEATURES = [
    "channel",
    "device",
    "plan",
    "country",
]

TARGET_CLF = "converted"
TARGET_REG = "revenue"

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15   # fracción del train para early stopping
RANDOM_STATE = 42
CV_FOLDS     = 5
SCORING      = "roc_auc"

# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------
N_TRIALS           = 60
EARLY_STOPPING_RDS = 40
OPTUNA_DB          = "sqlite:///optuna_study.db"
STUDY_NAME         = "xgb_conversion"

# Espacio de búsqueda XGBoost
XGB_PARAM_SPACE = {
    "learning_rate":    ("float_log", 1e-3, 0.3),
    "max_depth":        ("int",       3,    10),
    "subsample":        ("float",     0.5,  1.0),
    "colsample_bytree": ("float",     0.5,  1.0),
    "min_child_weight": ("int",       1,    20),
    "reg_alpha":        ("float_log", 1e-4, 10.0),
    "reg_lambda":       ("float_log", 1e-4, 10.0),
    "gamma":            ("float",     0.0,  2.0),
}
