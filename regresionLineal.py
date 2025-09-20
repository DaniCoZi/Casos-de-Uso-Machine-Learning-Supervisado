# -*- coding: utf-8 -*-
"""
Ingreso = f(Publicidad, Precio de venta)
Flujo: carga -> entrenamiento (OLS) -> predicción -> (gráfico 2D proyectado)
"""

import io, base64
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Paleta del sitio (dark)
DARK_BG = "#0b1220"
FG      = "#e5e7eb"
GRID    = "#1f2937"
ACCENT  = "#22d3ee"

def _apply_dark_mpl():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   DARK_BG,
        "savefig.facecolor":DARK_BG,
        "axes.edgecolor":   FG,
        "axes.labelcolor":  FG,
        "text.color":       FG,
        "xtick.color":      FG,
        "ytick.color":      FG,
        "axes.grid":        True,
        "grid.color":       GRID,
        "grid.alpha":       0.35,
        "legend.facecolor": "#0f172a",
        "legend.edgecolor": GRID,
        "legend.framealpha":0.85,
    })


# -----------------------------
# 1) Carga de datos
# Estructura esperada si usas CSV:
#   Publicidad, Precio, Ingreso
# -----------------------------
CSV_PATH = Path("data/ingresos_publicidad_precio.csv")

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
    # Asegúrate que las columnas coincidan:
    # Publicidad (float), Precio (float), Ingreso (float)
    X = df[["Publicidad", "Precio"]].values
    y = df["Ingreso"].values
    DATA_DESC = "Dataset propio cargado desde data/ingresos_publicidad_precio.csv."
else:
    # Dataset DEMO (sintético) para trilaterar el flujo
    rng = np.random.default_rng(7)
    n = 60
    publicidad = rng.uniform(0, 100, size=n)      # inversión en publicidad (unidades monetarias)
    precio = rng.uniform(10, 50, size=n)          # precio de venta
    # Generamos ingreso con una relación lineal + ruido
    # beta0=1000, beta1=25 (publicidad), beta2=-15 (precio)
    ingreso = 1000 + 25 * publicidad - 15 * precio + rng.normal(0, 120, size=n)

    df = pd.DataFrame({
        "Publicidad": publicidad,
        "Precio": precio,
        "Ingreso": ingreso
    })
    X = df[["Publicidad", "Precio"]].values
    y = df["Ingreso"].values
    DATA_DESC = "Dataset sintético: Ingreso ~ 25*Publicidad - 15*Precio + ruido + 1000."

# -----------------------------
# 2) Entrenamiento
# -----------------------------
model = LinearRegression()
model.fit(X, y)
y_hat = model.predict(X)

METRICS = {
    "r2": float(r2_score(y, y_hat)),
    "mae": float(mean_absolute_error(y, y_hat)),
    "mse": float(mean_squared_error(y, y_hat))
}

PARAMS = {
    "intercept": float(model.intercept_),            # β0
    "coef": [float(c) for c in model.coef_]         # [β1 (Publicidad), β2 (Precio)]
}

# -----------------------------
# 3) Predicción
# -----------------------------
def predict_from_values(values: Dict[str, float]) -> float:
    """
    values = {"Publicidad": <float>, "Precio": <float>}
    """
    try:
        inv = float(values["Publicidad"])
        price = float(values["Precio"])
    except Exception:
        raise ValueError("Se requieren valores numéricos para 'Publicidad' y 'Precio'.")
    x_arr = np.array([[inv, price]], dtype=float)
    return float(model.predict(x_arr)[0])

# Alias cómodo para las rutas
def predict_income(publicidad: float, precio: float) -> float:
    return predict_from_values({"Publicidad": publicidad, "Precio": precio})

# -----------------------------
# 4) Gráfico (base64)
# Como son 2 predictores, mostramos dos proyecciones 2D:
#   (a) Ingreso vs Publicidad (fijando Precio = mediana)
#   (b) Ingreso vs Precio (fijando Publicidad = mediana)
# Para la UI usamos (a).
# -----------------------------
_plot_cache = None
def make_regression_plot_base64(precio_fijo: float | None = None,
                                punto: tuple[float, float] | None = None) -> str:
    """
    Genera el gráfico en base64.
    - precio_fijo: si se pasa, la línea de regresión se proyecta con ese Precio;
                   si no, usa la mediana del dataset.
    - punto: (Publicidad, Precio) del formulario; si se pasa, se dibuja el punto
             predicho sobre la figura.
    """
    _apply_dark_mpl()  # tema oscuro coherente con tu UI

    # Precio usado para la proyección 2D
    if precio_fijo is None:
        precio_fijo = float(np.median(df["Precio"]))

    # Eje x: Publicidad; línea con Precio fijo
    x_pub = np.linspace(df["Publicidad"].min(), df["Publicidad"].max(), 200)
    x_line = np.column_stack([x_pub, np.full_like(x_pub, precio_fijo)])
    y_line = model.predict(x_line)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)

    # Dispersión original
    ax.scatter(df["Publicidad"], df["Ingreso"], alpha=0.85,
               label="Datos (Ingreso vs Publicidad)")

    # Línea de regresión (precio fijo)
    ax.plot(x_pub, y_line, linewidth=2.5, color=ACCENT,
            label=f"Regresión (Precio = {precio_fijo:.1f})")

    # Si hay valores del usuario, dibujar su punto predicho
    if punto is not None:
        pub_usr, precio_usr = float(punto[0]), float(punto[1])
        y_usr = float(model.predict(np.array([[pub_usr, precio_usr]], dtype=float))[0])
        ax.scatter([pub_usr], [y_usr], s=90, color="#38bdf8",
                   edgecolors="white", linewidths=1.4, zorder=5,
                   label="Tu predicción")
        # Marcador de referencia vertical
        ax.axvline(pub_usr, color="#38bdf8", alpha=0.25, linewidth=1)

    # Estética
    ax.set_xlabel("Publicidad")
    ax.set_ylabel("Ingreso")
    ax.set_title("Ingreso vs Publicidad (proyección 2D)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    fig.tight_layout(pad=1.2)

    # Exportar
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")



# -----------------------------
# 5) Utilidades para la UI
# -----------------------------
def get_model_params() -> Dict[str, float]:
    return {
        "intercept": round(PARAMS["intercept"], 4),
        "coef_publicidad": round(PARAMS["coef"][0], 4),
        "coef_precio": round(PARAMS["coef"][1], 4),
    }

def get_metrics() -> Dict[str, float]:
    return {k: round(v, 4) for k, v in METRICS.items()}

def get_data_description() -> str:
    return DATA_DESC
