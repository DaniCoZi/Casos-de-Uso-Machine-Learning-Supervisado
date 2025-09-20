import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ruta por defecto
CSV_DEFAULT_PATH = r"C:\Users\usuario\Documents\machine_learning\Master Web Studio\hotel_bookings.csv"

FEATURES_NUM = ["dias_antelacion", "estancia_previa", "tamano_grupo"]
FEATURES_CAT = ["tipo_habitacion"]
TARGET = "cancela"
CLASS_NAMES = ["No", "Sí"]

def generar_dataset_sintetico(path, n=500, seed=42):
    rng = np.random.default_rng(seed)
    dias_antelacion = rng.integers(0, 365, n)
    tipo_habitacion = rng.choice(["Standard", "Deluxe", "Suite"], size=n, p=[0.6, 0.3, 0.1])
    estancia_previa = rng.integers(0, 10, n)
    tamano_grupo = rng.integers(1, 6, n)

    prob_cancel = (
        0.20 + 0.001 * dias_antelacion
        + 0.05 * (tamano_grupo - 1)
        - 0.02 * estancia_previa
        + np.where(tipo_habitacion == "Suite", -0.10, 0.0)
    )
    prob_cancel = np.clip(prob_cancel, 0, 1)
    cancela = rng.binomial(1, prob_cancel)

    df = pd.DataFrame({
        "dias_antelacion": dias_antelacion,
        "tipo_habitacion": tipo_habitacion,
        "estancia_previa": estancia_previa,
        "tamano_grupo": tamano_grupo,
        "cancela": cancela
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df

def load_or_generate_df(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return generar_dataset_sintetico(csv_path)

def build_pipeline(df: pd.DataFrame):
    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[TARGET].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT)],
        remainder="passthrough"
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    return pipe, X_train, X_test, y_train, y_test

def evaluate_pipeline(model, X_test, y_test, fig_path="static/confusion_logit.png"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report_dict = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=4
    )
    report_df = pd.DataFrame(report_dict).T
    order = CLASS_NAMES + ["accuracy", "macro avg", "weighted avg"]
    report_df = report_df.reindex(order)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=[f"Real {c}" for c in CLASS_NAMES],
        columns=[f"Pred {c}" for c in CLASS_NAMES]
    )

    # Imagen
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de Confusión")
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES)
    plt.yticks(ticks, CLASS_NAMES)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()

    return {
        "accuracy": float(acc),
        "report_df": report_df,
        "confusion_matrix_df": cm_df
    }

def predict_label(model, features: dict, threshold=0.5):
    EXPECTED = FEATURES_NUM + FEATURES_CAT
    row = {k: features.get(k, None) for k in EXPECTED}
    x = pd.DataFrame([row], columns=EXPECTED)
    prob_si = float(model.predict_proba(x)[0, 1])
    label = "Sí" if prob_si >= float(threshold) else "No"
    return label, prob_si

# Alias para cumplir el nombre pedido en el enunciado
def evaluate(model, X_test, y_test, fig_path="static/confusion_logit.png"):
    return evaluate_pipeline(model, X_test, y_test, fig_path=fig_path)
