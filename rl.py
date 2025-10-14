import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================
# 1) Datos de ejemplo
#    (simulados: masa ≈ V*ρ + ruido)
# =========================
rng = np.random.default_rng(42)

# Volúmenes (m³) entre 0.005 y 0.05
volumen = rng.uniform(0.005, 0.05, 200)

# Densidades (kg/m³) de varios materiales (mezcla aleatoria)
densidades_posibles = np.array([500, 700, 1000, 2300, 2700, 7850])
densidad = rng.choice(densidades_posibles, size=200)

# Masa "real" con ruido (kg)
masa_real = volumen * densidad + rng.normal(0, 0.5, size=200)  # ruido ~ 0.5 kg

# =========================
# 2) Diseño de variables X
# =========================
interaccion = volumen * densidad
X = np.column_stack([volumen, densidad, interaccion])  # [V, ρ, V*ρ]
y = masa_real

# =========================
# 3) Split train/test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# =========================
# 4) Entrenar modelo
# =========================
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# =========================
# 5) Evaluar
# =========================
y_pred = modelo.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("Coeficientes [V, ρ, V*ρ]:", modelo.coef_)
print("Intercepto:", modelo.intercept_)
print(f"R²: {r2:.3f} | MAE: {mae:.3f} kg | RMSE: {rmse:.3f} kg")

# =========================
# 6) Línea base física
#    (sin entrenar, solo V*ρ)
# =========================
y_base = X_test[:, 2]  # V*ρ
r2_base = r2_score(y_test, y_base)
mae_base = mean_absolute_error(y_test, y_base)
rmse_base = mean_squared_error(y_test, y_base, squared=False)

print(f"[Base física V*ρ] R²: {r2_base:.3f} | MAE: {mae_base:.3f} kg | RMSE: {rmse_base:.3f} kg")

# =========================
# 7) Uso en producción
# =========================
def predecir_peso(volumen_m3, densidad_kg_m3):
    """Predice peso (kg) con el modelo entrenado."""
    x = np.array([[volumen_m3, densidad_kg_m3, volumen_m3 * densidad_kg_m3]])
    return float(modelo.predict(x)[0])

# Ejemplo
print("Predicción para V=0.02 m³, ρ=2700 kg/m³:",
      f"{predecir_peso(0.02, 2700):.2f} kg")
