import pandas as pd
import numpy as np
from datetime import datetime, timedelta

FUNNEL_STEPS = [
    "visita_landing",
    "inicio_solicitud",
    "datos_personales",
    "otp",
    "datos_financieros",
    "carga_documentos",
    "evaluacion_crediticia",
    "aprobacion",
    "firma_digital",
    "activacion_tarjeta",
]

# Conversion rates por paso (realistas para fintech)
STEP_CVR = {
    "visita_landing":          1.00,
    "inicio_solicitud":        0.35,
    "datos_personales":        0.75,
    "otp":                     0.68,
    "datos_financieros":       0.80,
    "carga_documentos":        0.60,
    "evaluacion_crediticia":   0.90,
    "aprobacion":              0.55,
    "firma_digital":           0.82,
    "activacion_tarjeta":      0.70,
}

# Multiplicadores por canal
CHANNEL_MULTIPLIERS = {
    "organic":  {"inicio_solicitud": 1.10, "otp": 1.05},
    "paid":     {"inicio_solicitud": 0.90, "aprobacion": 0.85},
    "email":    {"inicio_solicitud": 1.20, "datos_financieros": 1.10},
    "social":   {"inicio_solicitud": 0.80, "carga_documentos": 0.90},
    "direct":   {"inicio_solicitud": 1.15, "aprobacion": 1.05},
}

CHANNELS = list(CHANNEL_MULTIPLIERS.keys())


def generate_funnel_data(n_days=90, base_visits=3000, seed=42):
    np.random.seed(seed)
    records = []

    for day_offset in range(n_days):
        date = datetime(2024, 1, 1) + timedelta(days=day_offset)

        # Variación temporal (tendencia leve + ruido)
        trend = 1 + (day_offset / n_days) * 0.2
        dow_effect = 0.85 if date.weekday() >= 5 else 1.0

        for channel in CHANNELS:
            channel_share = {"organic": 0.35, "paid": 0.30, "email": 0.15, "social": 0.12, "direct": 0.08}
            visits = int(base_visits * channel_share[channel] * trend * dow_effect * np.random.uniform(0.85, 1.15))

            row = {"date": date.date(), "channel": channel}
            current = visits
            for step in FUNNEL_STEPS:
                cvr = STEP_CVR[step]
                # Aplicar multiplicador de canal si existe
                if step in CHANNEL_MULTIPLIERS[channel]:
                    cvr *= CHANNEL_MULTIPLIERS[channel][step]
                cvr = min(cvr, 1.0)
                current = int(current * cvr) if step != "visita_landing" else current
                row[step] = max(current, 0)

            records.append(row)

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_funnel_data()
    df.to_csv("data/raw/funnel_data.csv", index=False)
    print(f"Dataset generado: {df.shape}")
    print(df.head(10).to_string())
