# app.py
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, flash
import regresionLineal
import gbm_terrenos

# ─────────────── Flask ───────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# ─────────────── Rutas existentes ───────────────
@app.route("/")
def index():
    casos = [
        ("Detección de fraudes", "/caso/deteccion-fraudes"),
        ("Diagnóstico médico asistido", "/caso/diagnostico-medico"),
        ("Mantenimiento predictivo", "/caso/mantenimiento-predictivo"),
        ("Sistemas de recomendación", "/caso/recomendaciones"),
    ]
    return render_template("index.html", casos=casos)

@app.route("/caso/deteccion-fraudes")
def caso_fraudes():
    return render_template("deteccion_fraudes.html")

@app.route("/caso/diagnostico-medico")
def caso_diagnostico():
    return render_template("diagnostico_medico.html")

@app.route("/caso/spam")
def spam():
    return render_template("spam.html")

@app.route("/caso/reconocimiento-facial")
def reconocimiento_facial():
    return render_template("reconocimiento_facial.html")

@app.route("/rl/conceptos")
def rl_conceptos():
    return render_template("rl_conceptos.html")

# Ejercicio práctico (Regresión Lineal)
@app.route("/regresionLineal", methods=["GET", "POST"])
def RL():
    result = None
    vals = {"Publicidad": "", "Precio": ""}

    if request.method == "POST":
        try:
            inv = float(request.form["Publicidad"])
            precio = float(request.form["Precio"])
            vals["Publicidad"], vals["Precio"] = inv, precio

            result = regresionLineal.predict_income(inv, precio)
            plot_uri = regresionLineal.make_regression_plot_base64(
                precio_fijo=precio, punto=(inv, precio)
            )
        except Exception:
            result = None
            plot_uri = regresionLineal.make_regression_plot_base64()
    else:
        plot_uri = regresionLineal.make_regression_plot_base64()

    params  = regresionLineal.get_model_params()
    metrics = regresionLineal.get_metrics()
    data_desc = regresionLineal.get_data_description()

    return render_template(
        "rl.html",
        result=result,
        vals=vals,
        plot_uri=plot_uri,
        params=params,
        metrics=metrics,
        data_desc=data_desc
    )

# Pruebas (form simple, mismo modelo)
@app.route("/pruebasPracticas", methods=["GET", "POST"])
def pruebas():
    result = None
    if request.method == "POST":
        try:
            inv = float(request.form["Publicidad"])
            precio = float(request.form["Precio"])
            result = regresionLineal.predict_income(inv, precio)
        except Exception:
            result = None
    return render_template("pruebasPracticas.html", result=result)

# ─────────────── Regresión Logística (CSV backend + threshold dinámico) ───────────────
from regresionLogistica import (
    CSV_DEFAULT_PATH,
    load_or_generate_df,
    build_pipeline,
    predict_label,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Estado GBM
GBM_STATE = {
    "metrics": None,
    "last_prediction": None,
}

# Estado en memoria para la demo (Regresión Logística)
RLOGIT_STATE = {
    "model": None,
    "X_test": None,
    "y_test": None,
    "y_proba_test": None,
    "report_html": None,
    "cm_html": None,
    "accuracy": None,
    "threshold": 0.5,
    "csv_path": CSV_DEFAULT_PATH,
}

def _recompute_from_threshold(th: float, fig_path: str):
    y_test = RLOGIT_STATE["y_test"]
    y_proba = RLOGIT_STATE["y_proba_test"]
    if y_test is None or y_proba is None:
        raise RuntimeError("No hay probabilidades de test. Entrena el modelo primero.")

    th = float(max(0.0, min(1.0, th)))
    y_pred = (y_proba >= th).astype(int)

    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test, y_pred, target_names=["No", "Sí"], output_dict=True, digits=4
    )
    report_df = pd.DataFrame(report_dict).T
    order = ["No", "Sí", "accuracy", "macro avg", "weighted avg"]
    report_df = report_df.reindex(order)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm, index=["Real No", "Real Sí"], columns=["Pred No", "Pred Sí"]
    )

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de Confusión")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["No", "Sí"])
    plt.yticks(ticks, ["No", "Sí"])
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    thresh_val = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh_val else "black"
            )
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()

    RLOGIT_STATE["accuracy"] = float(acc)
    RLOGIT_STATE["report_html"] = report_df.to_html(
        classes="table table-hover table-sm align-middle caption-top tbl",
        float_format=lambda x: f"{x:0.4f}",
        border=0,
        index_names=False,
        justify="center",
    )
    RLOGIT_STATE["cm_html"] = cm_df.to_html(
        classes="table table-hover table-sm align-middle caption-top tbl",
        border=0,
        index_names=False,
        justify="center",
    )

@app.route("/rlogistica")
def rLogistica():
    return render_template(
        "rLogistico_practico.html",
        report_html=RLOGIT_STATE.get("report_html"),
        cm_html=RLOGIT_STATE.get("cm_html"),
        csv_path=RLOGIT_STATE.get("csv_path"),
        accuracy=RLOGIT_STATE.get("accuracy"),
        threshold=RLOGIT_STATE.get("threshold", 0.5),
    )

@app.route("/rlogistica/conceptos")
def rLogistica_conceptos():
    return render_template("rLogistico_conceptos.html")

@app.route("/rlogistica/entrenar", methods=["POST"])
def rLogistica_entrenar():
    RLOGIT_STATE["csv_path"] = CSV_DEFAULT_PATH
    try:
        df = load_or_generate_df(CSV_DEFAULT_PATH)
    except Exception as e:
        flash(f"Error cargando el dataset: {e}", "danger")
        return redirect(url_for("rLogistica"))

    try:
        pipe, X_train, X_test, y_train, y_test = build_pipeline(df)
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception as e:
        flash(f"Fallo en entrenamiento/evaluación: {e}", "danger")
        return redirect(url_for("rLogistica"))

    RLOGIT_STATE["model"] = pipe
    RLOGIT_STATE["X_test"] = X_test
    RLOGIT_STATE["y_test"] = y_test
    RLOGIT_STATE["y_proba_test"] = y_proba

    fig_path = os.path.join(STATIC_DIR, "confusion_logit.png")
    _recompute_from_threshold(RLOGIT_STATE["threshold"], fig_path)

    flash("Modelo entrenado y evaluado con el CSV del backend.", "success")
    return redirect(url_for("rLogistica"))

@app.route("/rlogistica/ajustar_umbral", methods=["POST"])
def rLogistica_ajustar_umbral():
    if RLOGIT_STATE["y_proba_test"] is None:
        flash("Primero entrena el modelo.", "warning")
        return redirect(url_for("rLogistica"))
    try:
        th = float(request.form.get("threshold", 0.5))
        th = max(0.0, min(1.0, th))
        RLOGIT_STATE["threshold"] = th
        fig_path = os.path.join(STATIC_DIR, "confusion_logit.png")
        _recompute_from_threshold(th, fig_path)
        flash(f"Umbral actualizado a {th:.2f}. Accuracy: {RLOGIT_STATE['accuracy']:.4f}", "info")
    except Exception as e:
        flash(f"No se pudo actualizar el umbral: {e}", "danger")
    return redirect(url_for("rLogistica"))

@app.route("/rlogistica/predecir", methods=["POST"])
def rLogistica_predecir():
    if RLOGIT_STATE["model"] is None:
        flash("Primero entrena el modelo.", "warning")
        return redirect(url_for("rLogistica"))

    try:
        dias_antelacion = int(request.form.get("dias_antelacion"))
        estancia_previa = int(request.form.get("estancia_previa"))
        tamano_grupo = int(request.form.get("tamano_grupo"))
        tipo_habitacion = request.form.get("tipo_habitacion")
        threshold = float(request.form.get("threshold", RLOGIT_STATE.get("threshold", 0.5)))

        ejemplo = {
            "dias_antelacion": dias_antelacion,
            "estancia_previa": estancia_previa,
            "tamano_grupo": tamano_grupo,
            "tipo_habitacion": tipo_habitacion,
        }
        label, prob = predict_label(RLOGIT_STATE["model"], ejemplo, threshold=threshold)
        cat = "success" if label == "No" else "warning"
        flash(f"Predicción: {label} (prob de 'Sí' = {prob:.3f})", cat)
    except Exception as e:
        flash(f"Error en la predicción: {e}", "danger")

    return redirect(url_for("rLogistica"))

@app.route('/TiposAlgoritClasi', endpoint='tipos_algorit_clasi')
def tipos_algorit_clasi():
    return render_template('tiposAlgoritClasi.html')

# ─────────────── GBM: helpers ───────────────
def _norm(x):
    """Acepta '0,5' o '0.5'."""
    return str(x).replace(',', '.').strip()

# Página del caso práctico GBM
@app.route("/clasificacion/caso-practico/gbm", methods=["GET"])
def gbm_caso():
    return render_template(
        "/gbm_caso_practico.html",
        metrics=GBM_STATE.get("metrics"),
        prediction=GBM_STATE.get("last_prediction"),
    )

# Entrenar y evaluar GBM (acepta GET y POST para evitar 405 si alguien navega directo)
@app.route("/clasificacion/caso-practico/gbm/evaluar", methods=["GET", "POST"])
def gbm_evaluar():
    try:
        metrics = gbm_terrenos.evaluate(static_dir=STATIC_DIR)
        GBM_STATE["metrics"] = metrics
        flash(f"GBM evaluado. Accuracy: {metrics['accuracy']:.4f}", "success")
    except Exception as e:
        flash(f"Error al evaluar GBM: {e}", "danger")
    return redirect(url_for("gbm_caso"))

# Predicción
@app.route("/clasificacion/caso-practico/gbm/predecir", methods=["POST"])
def gbm_predecir():
    try:
        payload = {
            "tipo_suelo": request.form.get("tipo_suelo"),
            "altitud": _norm(request.form.get("altitud")),
            "humedad": _norm(request.form.get("humedad")),
            "prox_agua": _norm(request.form.get("prox_agua")),
            "uso_circundante": request.form.get("uso_circundante"),
        }
        th = float(_norm(request.form.get("threshold", 0.5)))
        pred = gbm_terrenos.predict_label(payload, threshold=th)
        GBM_STATE["last_prediction"] = pred

        conf_note = " (baja confianza)" if pred.get("low_confidence") else ""
        flash(f"Predicción GBM: {pred['label']} — max p={max(pred['probas'].values()):.3f}{conf_note}", "info")
    except Exception as e:
        flash(f"Error en la predicción GBM: {e}", "danger")
    return redirect(url_for("gbm_caso"))

# ─────────────── Main ───────────────
if __name__ == "__main__":
    app.run(debug=True)
