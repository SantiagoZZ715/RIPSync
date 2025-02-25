import pandas as pd
import numpy as np
import joblib
import json
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os    
from difflib import get_close_matches

# Configuración de archivos
MODELO_ENTRENADO = r"Modelos_entrenados\modelo_random_forest_RIPS_consulta.pkl"
ARCHIVO_BASE_ACTUALIZADA = r"Bases_de_conocimiento\base_actualizada.xlsx"

# Variables que NO se validarán
EXCLUIR_VALIDACION = [
    "numFEVPagoModerador", "valorPagoModerador", "vrServicio",
    "numDocumentoldentificacion", "numAutorizacion", "fechalnicioAtencion", "consecutivo"
]

# Variables que pueden estar vacías sin ser incorrectas
COLUMNAS_OPCIONALMENTE_NULAS = {"codDiagnosticoRelacionador", "codDiagnosticoRelacionado2", "codDiagnosticoRelacionado3"}

# Diccionario para almacenar valores válidos
VALORES_VALIDOS = {}

# Función para cargar la base de datos asegurando que siempre incluya la original
def cargar_base_datos():
    if os.path.exists(ARCHIVO_BASE_ACTUALIZADA):
        df_actualizada = pd.read_excel(ARCHIVO_BASE_ACTUALIZADA, dtype=str)
    else:
        df_actualizada = pd.DataFrame()

    df_original = pd.read_excel(r"Bases_de_conocimiento\datos_correctos.xlsx", dtype=str)

    # Fusionar bases de datos asegurando que los datos correctos no se pierdan
    df = pd.concat([df_original, df_actualizada], ignore_index=True).drop_duplicates(subset=df_original.columns)

    # Extraer valores únicos correctos por cada columna
    for columna in df.columns:
        if columna not in EXCLUIR_VALIDACION:
            VALORES_VALIDOS[columna] = set(df[columna].dropna().astype(str))

    return df

# Cargar base de referencia
df = cargar_base_datos()

# Verificar si la columna "valid" existe antes de eliminarla
X = df.drop(columns=["valid"]) if "valid" in df.columns else df.copy()

# Identificar columnas categóricas
columnas_categoricas = X.select_dtypes(include=['object']).columns
umbral_categorias = 1000

# Separar columnas categóricas en pocas y muchas categorías
columnas_a_codificar = [col for col in columnas_categoricas if X[col].nunique() < umbral_categorias]
columnas_a_dejar = [col for col in columnas_categoricas if X[col].nunique() >= umbral_categorias]

# Aplicar codificación adecuada
X_dummies = pd.get_dummies(X[columnas_a_codificar])
for col in columnas_a_dejar:
    X[col], _ = X[col].factorize()

X = pd.concat([X.drop(columns=columnas_categoricas), X_dummies], axis=1)

# Dividir datos para entrenar el modelo
X_train, X_test, y_train, y_test = train_test_split(X, np.ones(len(X)), test_size=0.2, random_state=42)

# Cargar modelo si existe, si no, entrenarlo
if os.path.exists(MODELO_ENTRENADO):
    clf = joblib.load(MODELO_ENTRENADO)
    print("Modelo cargado desde archivo.")
else:
    print("No se encontró un modelo guardado. Entrenando desde cero...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODELO_ENTRENADO)
    print("Modelo entrenado y guardado correctamente.")

# Función para verificar el formato de fecha
def validar_formato_fecha(fecha):
    try:
        datetime.datetime.strptime(fecha, "%Y-%m-%d %H:%M")
        return True
    except ValueError:
        return False

# Función para encontrar valores similares
def encontrar_valores_mas_cercanos(valor, columna, max_sugerencias=3):
    valores_correctos = list(VALORES_VALIDOS.get(columna, []))
    if not valores_correctos:
        return None

    coincidencias = get_close_matches(str(valor), valores_correctos, n=max_sugerencias, cutoff=0.5)
    return coincidencias if coincidencias else None

# Función para validar JSON
def validar_json(nombre_archivo):
    with open(nombre_archivo, "r", encoding="utf-8") as file:
        datos_usuario = json.load(file)

    resultados = []
    datos_corregidos = []  # Almacena los datos con correcciones
    nuevos_datos = []

    for registro in datos_usuario:
        consecutivo = registro.get("consecutivo", "Desconocido")
        errores = {}
        registro_corregido = registro.copy()  # Copia del registro para corregir valores

        # Convertir todos los valores a string para evitar errores de comparación
        registro = {key: str(value) if value is not None else "" for key, value in registro.items()}

        # Validar formato de fecha
        if "fechalnicioAtencion" in registro and not validar_formato_fecha(registro["fechalnicioAtencion"]):
            errores["fechalnicioAtencion"] = "Formato incorrecto (Esperado: YYYY-MM-DD HH:MM)"
            registro_corregido["fechalnicioAtencion"] = "CORREGIR_FECHA"

        # Validar cada columna asegurando que la comparación sea entre strings
        for columna, valores_correctos in VALORES_VALIDOS.items():
            valores_correctos_str = set(map(str, valores_correctos))

            if columna in registro:
                valor_usuario = str(registro[columna])

                # Si la columna puede estar vacía y el usuario no ingresó nada, la dejamos pasar
                if columna in COLUMNAS_OPCIONALMENTE_NULAS and valor_usuario in ["", "None", "null"]:
                    continue  

                if valor_usuario not in valores_correctos_str:
                    valores_sugeridos = encontrar_valores_mas_cercanos(valor_usuario, columna, max_sugerencias=3)
                    errores[columna] = {
                        "valor ingresado": valor_usuario,
                        "sugerencias": valores_sugeridos if valores_sugeridos else "No se encontraron valores similares"
                    }

                    # Aplicar la primera sugerencia si existe, sino dejar el valor original
                    if valores_sugeridos:
                        registro_corregido[columna] = valores_sugeridos[0]
                    else:
                        registro_corregido[columna] = f"REVISAR_{valor_usuario}"

        # Guardar registro corregido
        datos_corregidos.append(registro_corregido)

        if not errores:
            resultados.append({"consecutivo": consecutivo, "estado": "Correcto"})
            nuevos_datos.append(registro)
        else:
            resultados.append({"consecutivo": consecutivo, "estado": "Incorrecto", "errores": errores})

    # Guardar JSON con correcciones sugeridas
    with open(r"Datos_corregidos\datos_corregidos.json", "w", encoding="utf-8") as outfile:
        json.dump(datos_corregidos, outfile, indent=4)

    print(r"Archivo 'Datos_corregidos\datos_corregidos.json' generado con sugerencias de corrección.")

    if nuevos_datos:
        guardar_nuevos_datos(nuevos_datos)
        reentrenar_modelo()

    return resultados

# Función para guardar nuevos datos válidos
def guardar_nuevos_datos(nuevos_datos):
    df_nuevos = pd.DataFrame(nuevos_datos)
    if os.path.exists(ARCHIVO_BASE_ACTUALIZADA):
        df_base = pd.read_excel(ARCHIVO_BASE_ACTUALIZADA)
        df_base = pd.concat([df_base, df_nuevos], ignore_index=True)
    else:
        df_base = df_nuevos
    df_base.to_excel(ARCHIVO_BASE_ACTUALIZADA, index=False)
    print("Nuevos datos añadidos para aprendizaje continuo.")

# Función para reentrenar el modelo
def reentrenar_modelo():
    global clf
    print("Reentrenando el modelo...")
    df = cargar_base_datos()
    X = df.drop(columns=["valid"]) if "valid" in df.columns else df.copy()
    
    columnas_categoricas = X.select_dtypes(include=['object']).columns
    columnas_a_codificar = [col for col in columnas_categoricas if X[col].nunique() < 1000]
    columnas_a_dejar = [col for col in columnas_categoricas if X[col].nunique() >= 1000]

    X_dummies = pd.get_dummies(X[columnas_a_codificar])
    for col in columnas_a_dejar:
        X[col], _ = X[col].factorize()

    X = pd.concat([X.drop(columns=columnas_categoricas), X_dummies], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, np.ones(len(X)), test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODELO_ENTRENADO)
    print(" Modelo reentrenado y guardado.")
    

