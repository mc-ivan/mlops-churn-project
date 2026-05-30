from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
TEST_DATA = DATA_DIR / "test.csv"
MODELS = {
    "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
    "Random Forest": MODELS_DIR / "random_forest.pkl"
}
METRICS_FILE = DOCS_DIR / "model_metrics.md"

def evaluar_modelo():
    """
    Evalúa el modelo entrenado y guarda las métricas principales.
    """

    if not TEST_DATA.exists():
        raise FileNotFoundError(
            "No se encontró data/test.csv. Primero ejecuta src/preprocess.py"
        )

    if not MODELS["Logistic Regression"].exists():
        raise FileNotFoundError(
            "No se encontró el modelo Logistic Regression entrenado. Primero ejecuta src/train_model.py"
        )
    
    if not MODELS["Random Forest"].exists():
        raise FileNotFoundError(
            "No se encontró el modelo Random Forest entrenado. Primero ejecuta src/train_model.py"
        )

    DOCS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(TEST_DATA)

    X_test = df.drop(columns=["churn"])
    y_test = df["churn"]

    resultados = []
    for nombre, archivo_modelo in MODELS.items():

        modelo = joblib.load(archivo_modelo)

        y_pred = modelo.predict(X_test)

        resultados.append({
            "modelo": nombre,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0)
        })

    contenido = """# Churn Model Metrics

## Comparación de modelos

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
"""
    for resultado in resultados:
        contenido += (
            f"| {resultado['modelo']} "
            f"| {resultado['accuracy']:.4f} "
            f"| {resultado['precision']:.4f} "
            f"| {resultado['recall']:.4f} "
            f"| {resultado['f1']:.4f} |\n"
        )

    contenido += f"""\n\n## Interpretación inicial

Estas métricas permiten evaluar el desempeño inicial del modelo de clasificación.

- Accuracy indica el porcentaje general de aciertos.
- Precision indica qué tan confiables son las predicciones positivas.
- Recall indica qué proporción de clientes con churn fueron identificados.
- F1-score resume precision y recall en una sola métrica.
"""

    METRICS_FILE.write_text(contenido, encoding="utf-8")

    print("Modelos evaluados correctamente.")
    print(f"Métricas guardadas en: {METRICS_FILE}")

if __name__ == "__main__":
    evaluar_modelo()