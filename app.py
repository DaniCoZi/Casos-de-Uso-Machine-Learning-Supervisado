from flask import Flask, render_template, request


app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(debug=True)
