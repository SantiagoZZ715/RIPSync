import os
import json
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from Modelos.Modelo_RIPS_Consulta import validar_json  # Importamos la función de validación

app = Flask(__name__)

# Configuración de rutas de archivos
UPLOAD_FOLDER = "Documentos_de_entrada"
RESULT_FOLDER = "Errores_identificados"
CORRECTED_FOLDER = "Datos_corregidos"
ALLOWED_EXTENSIONS = {"json"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["CORRECTED_FOLDER"] = CORRECTED_FOLDER
app.secret_key = "supersecretkey"

# Crear carpetas si no existen
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, CORRECTED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

modelo = joblib.load(r"Modelos_entrenados\modelo_cups\modelo_cups_v2.pkl")
vectorizer = joblib.load(r"Modelos_entrenados\modelo_cups\vectorizer_v2.pkl")
le = joblib.load(r"Modelos_entrenados\modelo_cups\label_encoder_v2.pkl")

df = pd.read_excel(r"Documentos_procesados\cups_procesado.xlsx")
df["Habilitado"] = df["Habilitado"].apply(lambda x: 1 if x == "SI" else 0)
df["UsoCodigoCUP_encoded"] = df["UsoCodigoCUP"].map({"AP": 1, "AC": 0})

def validar_cups(cups_ingresado, descripcion, tipo_atencion):
    if cups_ingresado not in df["Codigo"].values:
        desc_vectorizada = vectorizer.transform([descripcion]).toarray()
        desc_completa = vectorizer.transform(df["Nombre"]).toarray()
        similitudes = cosine_similarity(desc_vectorizada, desc_completa)
        idx_mas_similar = similitudes.argmax()
        cups_sugerido = df.iloc[idx_mas_similar]["Codigo"]
        descripcion_sugerida = df.iloc[idx_mas_similar]["Nombre"]
        tipo_atencion_sugerido = df.iloc[idx_mas_similar]["UsoCodigoCUP"]
        return f"❌ CUPS incorrecto. Sugerencia: Usa {cups_sugerido} - {descripcion_sugerida} ({tipo_atencion_sugerido})."

    datos_cups = df[df["Codigo"] == cups_ingresado].iloc[0]
    tipo_atencion_correcto = datos_cups["UsoCodigoCUP"]
    tipo_atencion_ingresado = 1 if tipo_atencion == "AP" else 0

    if tipo_atencion_ingresado != datos_cups["UsoCodigoCUP_encoded"]:
        tipo_correcto_text = "AP" if datos_cups["UsoCodigoCUP_encoded"] == 1 else "AC"
        ejemplo = df[df["UsoCodigoCUP_encoded"] == datos_cups["UsoCodigoCUP_encoded"]].iloc[0]
        return f"❌ Tipo de atención incorrecto. El CUPS {cups_ingresado} es para '{tipo_correcto_text}'. Ejemplo: Usa {ejemplo['Codigo']} - {ejemplo['Nombre']}."

    descripcion_correcta = datos_cups["Nombre"]
    desc_vectorizada = vectorizer.transform([descripcion]).toarray()
    desc_correcta_vectorizada = vectorizer.transform([descripcion_correcta]).toarray()
    similitud = cosine_similarity(desc_vectorizada, desc_correcta_vectorizada)[0][0]

    if similitud < 0.8:
        return f"❌ La descripción no coincide con el CUPS ingresado. El CUPS {cups_ingresado} corresponde a: '{descripcion_correcta}'."

    return "✅ El CUPS ingresado, la descripción y el tipo de atención son válidos."

@app.route("/", methods=["GET", "POST"])
def index():
    validaciones = None
    archivo_corregido = False
    correctos = 0
    incorrectos = 0

    if request.method == "POST":
        if "file" not in request.files:
            flash("No se seleccionó ningún archivo", "error")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No se seleccionó ningún archivo", "error")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Validar el JSON
            resultado = validar_json(filepath)

            # Contar correctos e incorrectos
            correctos = sum(1 for item in resultado if item["estado"] == "Correcto")
            incorrectos = sum(1 for item in resultado if item["estado"] == "Incorrecto")

            # Guardar resultado en JSON
            resultado_path = os.path.join(app.config["RESULT_FOLDER"], "resultado_validacion.json")
            with open(resultado_path, "w", encoding="utf-8") as f:
                json.dump(resultado, f, indent=4)

            # Revisar si hay datos corregidos
            corrected_file = os.path.join(app.config["CORRECTED_FOLDER"], "datos_corregidos.json")
            archivo_corregido = os.path.exists(corrected_file) and incorrectos > 0  # Solo si hay errores

            return render_template(
                "index.html",
                validaciones=resultado,
                archivo_corregido=archivo_corregido,
                correctos=correctos,
                incorrectos=incorrectos
            )

    return render_template("index.html", validaciones=None, correctos=0, incorrectos=0)

@app.route("/descargar")
def descargar():
    corrected_file = os.path.join(app.config["CORRECTED_FOLDER"], "datos_corregidos.json")
    if os.path.exists(corrected_file):
        return send_file(corrected_file, as_attachment=True)
    flash("No hay archivo corregido disponible", "error")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
