"""
features.py — Construcción del Pipeline de preprocesamiento sklearn.

Separar el preprocesador del modelo permite:
  - Reusar el mismo transform en train / predict sin reentrenar.
  - Inspeccionar feature names después del encoding.
  - Intercambiar modelos sin tocar el preprocesamiento.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from config import CAT_FEATURES, NUM_FEATURES


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """
    Devuelve un ColumnTransformer listo para fit/transform.

    Args:
        scale_numeric: Si True, aplica StandardScaler sobre las numéricas.
                       Para GBTs no es necesario; útil para modelos lineales.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), NUM_FEATURES),
            ("cat", cat_pipeline,        CAT_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_full_pipeline(model, scale_numeric: bool = False) -> Pipeline:
    """
    Envuelve preprocesador + modelo en un sklearn Pipeline.
    Permite pasar directamente DataFrames sin preprocesar.
    """
    return Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
        ("model",        model),
    ])


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Devuelve los nombres de features tras el transform (post-fit)."""
    return list(preprocessor.get_feature_names_out())
