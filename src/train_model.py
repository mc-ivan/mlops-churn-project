from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TRAIN_DATA = DATA_DIR / "train.csv"
LOGISTIC_MODEL_FILE = MODELS_DIR / "logistic_regression.pkl"
RANDOM_FOREST_MODEL_FILE = MODELS_DIR / "random_forest.pkl"

def entrenar_modelo():
    """
    Entrena un modelo simple de clasificación para predecir churn.
    """
    if not TRAIN_DATA.exists():
        raise FileNotFoundError(
            "No se encontró data/train.csv. Primero ejecuta src/preprocess.py"
        )

    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(TRAIN_DATA)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Logistic Regression Model
    logistic_model = Pipeline(
        steps=[
            ("escalado", StandardScaler()),
            ("clasificador", LogisticRegression())
        ]
    )

    logistic_model.fit(X, y)

    joblib.dump(
        logistic_model,
        LOGISTIC_MODEL_FILE
    )

    print(f"Modelo guardado en: {LOGISTIC_MODEL_FILE}")

    # Random Forest Model
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    random_forest_model.fit(X, y)

    joblib.dump(
        random_forest_model,
        RANDOM_FOREST_MODEL_FILE
    )

    print(f"Modelo guardado en: {RANDOM_FOREST_MODEL_FILE}")
    print("Modelos entrenados correctamente.")

if __name__ == "__main__":
    entrenar_modelo()
