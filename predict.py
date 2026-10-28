"""Inferencia y validación compartidas por la interfaz y las pruebas."""
import math
import joblib
import numpy as np
import pandas as pd
from config import ROOT, FEATURES, LABELS


def validate_record(record):
    errors = []
    if set(record) != set(FEATURES):
        return ["Faltan variables o se recibieron campos inesperados."]
    try:
        values = {key: float(record[key]) for key in FEATURES}
    except (TypeError, ValueError):
        return ["Todos los campos deben ser numéricos."]
    if not all(math.isfinite(v) for v in values.values()):
        return ["No se permiten campos vacíos, infinitos ni valores no numéricos."]
    if not 17 <= values[FEATURES[0]] <= 70:
        errors.append("La edad debe estar entre 17 y 70 años, rango observado en la fuente.")
    for feature in FEATURES[1:3]:
        if not 0 <= values[feature] <= 200:
            errors.append("Las notas de ingreso deben usar la escala original de 0 a 200.")
    if not 0 <= values[FEATURES[7]] <= 20:
        errors.append("El promedio debe usar la escala original de 0 a 20.")
    for key in [FEATURES[0], *FEATURES[3:7], FEATURES[8]]:
        if values[key] < 0 or not values[key].is_integer():
            errors.append("La edad y los conteos deben ser enteros no negativos.")
            break
    if values[FEATURES[6]] > values[FEATURES[4]]:
        errors.append("Las asignaturas aprobadas no pueden superar las inscritas.")
    if values[FEATURES[3]] > values[FEATURES[4]]:
        errors.append("Las asignaturas reconocidas no pueden superar las inscritas.")
    if values[FEATURES[8]] > values[FEATURES[4]]:
        errors.append("Las asignaturas sin evaluación no pueden superar las inscritas.")
    if (values[FEATURES[6]] == 0) != (values[FEATURES[7]] == 0):
        errors.append("Si no hay asignaturas aprobadas, el promedio de aprobadas debe ser 0; si hay aprobadas, debe ser mayor que 0.")
    for key, maximum in [(FEATURES[3],20), (FEATURES[4],26), (FEATURES[5],45), (FEATURES[6],26), (FEATURES[8],12)]:
        if values[key] > maximum:
            errors.append("Uno de los conteos supera el máximo observado en el dataset.")
            break
    return errors


def predict_record(model, record):
    errors = validate_record(record)
    if errors:
        raise ValueError(" ".join(errors))
    X = pd.DataFrame([record], columns=FEATURES)
    probabilities = model.predict_proba(X)[0]
    predicted = model.classes_[int(np.argmax(probabilities))]
    return {"clase": str(predicted), "resultado": LABELS[str(predicted)],
            "probabilidades": {str(c): float(p) for c, p in zip(model.classes_, probabilities)}}


def load_model():
    return joblib.load(ROOT / "artifacts/model.joblib")
