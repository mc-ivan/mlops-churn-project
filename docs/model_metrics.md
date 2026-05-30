# Churn Model Metrics

## Comparación de modelos

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |


## Interpretación inicial

Estas métricas permiten evaluar el desempeño inicial del modelo de clasificación.

- Accuracy indica el porcentaje general de aciertos.
- Precision indica qué tan confiables son las predicciones positivas.
- Recall indica qué proporción de clientes con churn fueron identificados.
- F1-score resume precision y recall en una sola métrica.


## Matrices de Confusión


### Logistic Regression

| | Predicho 0 | Predicho 1 |
|---|---:|---:|
| Real 0 | 3 | 0 |
| Real 1 | 0 | 2 |


### Random Forest

| | Predicho 0 | Predicho 1 |
|---|---:|---:|
| Real 0 | 3 | 0 |
| Real 1 | 0 | 2 |
