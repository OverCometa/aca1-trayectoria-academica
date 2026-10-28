"""Definición del experimento, fijada antes de consultar el conjunto de prueba."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42
CLASSES = ["Dropout", "Enrolled", "Graduate"]
LABELS = {"Dropout": "Abandono", "Enrolled": "Continúa matriculado", "Graduate": "Graduación"}
FEATURES = [
    "Age at enrollment", "Admission grade", "Previous qualification (grade)",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
]
FEATURE_LABELS = {
    FEATURES[0]: "Edad al ingresar", FEATURES[1]: "Nota de admisión",
    FEATURES[2]: "Nota de formación previa", FEATURES[3]: "Asignaturas reconocidas",
    FEATURES[4]: "Asignaturas inscritas", FEATURES[5]: "Intentos de evaluación",
    FEATURES[6]: "Asignaturas aprobadas", FEATURES[7]: "Promedio de aprobadas",
    FEATURES[8]: "Asignaturas sin evaluación",
}
DATA_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
DATA_DOI = "https://doi.org/10.24432/C5MC89"
TITLE = "Predicción de la trayectoria académica al finalizar el primer semestre"
AUTHOR = "Over Santiago Cometa Hernández"
