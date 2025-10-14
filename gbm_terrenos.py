# gbm_terrenos.py
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # seguro en servidores sin GUI
import matplotlib.pyplot as plt

from typing import Dict
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---- Configuración y paths ----
SEED = 42
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "terrenos.csv")

# Clases del problema (multiclase)
CLASSES = ["Residencial", "Industrial", "Agrícola"]

@dataclass
class GBMArtifacts:
    pipe: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    label_names: list

_STATE: Dict[str, object] = {
    "artifacts": None,          # GBMArtifacts
    "metrics": None,            # dict (evaluate)
    "images": {}                # rutas de imágenes generadas
}

# -----------------------------
# Datos: cargar o generar
# -----------------------------
def load_or_generate_df(path: str = CSV_PATH) -> pd.DataFrame:
    """
    Carga un CSV si existe. Si no, genera un dataset sintético multiclase
    con:
      - tipo_suelo (cat)
      - altitud (num)
      - humedad (num 0-1)
      - prox_agua (num, metros)
      - uso_circundante (cat)
      - uso_recomendado (label: Residencial/Industrial/Agrícola)
    """
    if os.path.exists(path):
        return pd.read_csv(path)

    rng = np.random.default_rng(SEED)
    n = 700

    tipo_suelo = rng.choice(
        ["franco", "arcilloso", "arenoso", "limoso"],
        size=n, p=[0.4, 0.25, 0.25, 0.10]
    )
    altitud = rng.normal(900, 350, size=n).clip(0)  # metros
    humedad = rng.beta(2, 3, size=n)                # 0..1
    prox_agua = rng.lognormal(mean=6.2, sigma=0.7, size=n).clip(10, 15000)  # metros
    uso_circ = rng.choice(
        ["residencial", "industrial", "agrícola", "mixto"],
        size=n, p=[0.35, 0.25, 0.25, 0.15]
    )

    # Reglas latentes (vectorizadas con NumPy; antes usaba .between de pandas)
    score_res = (
        (prox_agua < 2000).astype(int)
        + (((humedad >= 0.25) & (humedad <= 0.65)).astype(int))
        + (pd.Series(uso_circ).isin(["residencial", "mixto"]).astype(int))
    )
    score_ind = (
        (prox_agua > 4000).astype(int)
        + (altitud < 1200).astype(int)
        + (pd.Series(uso_circ).isin(["industrial"]).astype(int))
    )
    score_agr = (
        (humedad > 0.55).astype(int)
        + (((altitud >= 600) & (altitud <= 1400)).astype(int))
        + (pd.Series(uso_circ).isin(["agrícola", "mixto"]).astype(int))
    )

    scores = np.vstack([score_res, score_ind, score_agr]).T
    label_idx = scores.argmax(axis=1)
    uso_recomendado = [CLASSES[i] for i in label_idx]

    df = pd.DataFrame({
        "tipo_suelo": tipo_suelo,
        "altitud": np.round(altitud, 1),
        "humedad": np.round(humedad, 3),
        "prox_agua": np.round(prox_agua, 0),
        "uso_circundante": uso_circ,
        "uso_recomendado": uso_recomendado,
    })
    df.to_csv(path, index=False)
    return df

# -----------------------------
# Pipeline sin leakage
# -----------------------------
def build_pipeline(df: pd.DataFrame) -> GBMArtifacts:
    y = df["uso_recomendado"]
    X = df.drop(columns=["uso_recomendado"])

    cat_cols = ["tipo_suelo", "uso_circundante"]
    num_cols = ["altitud", "humedad", "prox_agua"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]), num_cols),
        ],
        remainder="drop"
    )

    gbm = GradientBoostingClassifier(
        random_state=SEED,
        n_estimators=180,
        learning_rate=0.08,
        max_depth=3,
        subsample=0.9
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", gbm)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    pipe.fit(X_train, y_train)
    return GBMArtifacts(pipe=pipe, X_test=X_test, y_test=y_test, label_names=CLASSES)

# -----------------------------
# Evaluación y gráficos
# -----------------------------
def _save_confusion(cm: np.ndarray, labels: list, out_path: str) -> None:
    plt.figure(figsize=(5.4, 4.6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de Confusión")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

def evaluate(static_dir: str) -> Dict:
    """
    Entrena (o reentrena) GBM con split 80/20, calcula métricas y guarda imagen de confusión.
    Retorna dict con accuracy, per_class (precision/recall/f1/support), matrix e imágenes.
    """
    df = load_or_generate_df(CSV_PATH)
    artifacts = build_pipeline(df)

    pipe, X_test, y_test = artifacts.pipe, artifacts.X_test, artifacts.y_test

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test, y_pred, labels=CLASSES, output_dict=True, digits=4
    )
    per_class = {
        c: {
            "precision": report[c]["precision"],
            "recall": report[c]["recall"],
            "f1": report[c]["f1-score"],
            "support": int(report[c]["support"]),
        } for c in CLASSES
    }

    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)

    # Imagen
    conf_path = os.path.join(static_dir, "gbm_confusion.png")
    _save_confusion(cm, CLASSES, conf_path)

    metrics = {
        "accuracy": float(acc),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "images": {"confusion": "/static/gbm_confusion.png"},
    }

    _STATE["artifacts"] = artifacts
    _STATE["metrics"] = metrics
    _STATE["images"]["confusion"] = conf_path
    return metrics

# -----------------------------
# Predicción unitaria
# -----------------------------
def predict_label(features: Dict, threshold: float = 0.5) -> Dict:
    """
    Predice etiqueta en CLASSES con proba por clase.
    'threshold' se usa como umbral de confianza en multiclase: low_confidence=True
    si max(proba) < threshold.
    """
    if _STATE.get("artifacts") is None:
        _STATE["metrics"] = evaluate(static_dir=os.path.join(BASE_DIR, "static"))

    art: GBMArtifacts = _STATE["artifacts"]  # type: ignore
    pipe = art.pipe

    X = pd.DataFrame([{
        "tipo_suelo": features.get("tipo_suelo"),
        "altitud": float(features.get("altitud")),
        "humedad": float(features.get("humedad")),
        "prox_agua": float(features.get("prox_agua")),
        "uso_circundante": features.get("uso_circundante"),
    }])

    probas = pipe.predict_proba(X)[0]
    class_order = list(pipe.classes_)
    proba_map = {cls: float(p) for cls, p in zip(class_order, probas)}

    label = class_order[int(np.argmax(probas))]
    max_p = float(np.max(probas))
    low_conf = max_p < float(threshold)

    return {
        "label": label,
        "probas": proba_map,
        "low_confidence": low_conf,
        "threshold": float(threshold)
    }
