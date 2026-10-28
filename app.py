"""Interfaz académica de Streamlit integrada con el pipeline entrenado."""
import json
import pandas as pd
import streamlit as st
from config import ROOT, FEATURES, FEATURE_LABELS, LABELS, AUTHOR
from predict import load_model, predict_record, validate_record

st.set_page_config(page_title="Trayectoria académica | ACA 1", page_icon="🎓", layout="wide")

@st.cache_resource
def resources():
    return load_model(), json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))

model, metrics = resources()
st.markdown("<style>.block-container{max-width:1180px;padding-top:4rem} h1{font-size:2.25rem!important} div[data-testid='stMetric']{background:#eef5f7;padding:14px;border-radius:10px} .stButton>button{border-radius:8px} </style>", unsafe_allow_html=True)
st.caption("CUN · APRENDIZAJE AUTOMÁTICO · ACA 1 · PILAR EDUCACIÓN")
st.title("Trayectoria académica estudiantil")
st.markdown("Estima el resultado académico final a partir de información de ingreso y del **primer semestre**.")
st.caption(f"{AUTHOR} · Trabajo individual · Prototipo académico con datos de Portugal")

with st.sidebar:
    st.header("Explorar el proyecto")
    st.markdown("<style>div[data-testid='stMetric'], div[data-testid='stMetric'] * {color: #132c40 !important;}</style>", unsafe_allow_html=True)
    page = st.radio("Sección", ["Simular un caso", "Resultados del modelo", "Datos y metodología", "Referencias"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Momento de uso**\n\nAl cerrar el primer semestre.")
    st.markdown("**Tres resultados**\n\nAbandono · Continúa matriculado · Graduación")
    st.caption("Las etiquetas corresponden al final de la duración normal del programa, no al resultado inmediato del semestre.")

if page == "Simular un caso":
    st.subheader("Ingresa un caso de ejemplo")
    st.info("Usa las escalas originales: notas de ingreso de 0 a 200 y promedio de aprobadas de 0 a 20. No ingreses nombres ni documentos de identidad.")
    examples = {
        "Personalizado": [20, 125., 130., 0, 6, 8, 5, 12., 0],
        "Ejemplo A · buen avance": [20, 140., 145., 0, 6, 6, 6, 15., 0],
        "Ejemplo B · dificultades académicas": [24, 110., 120., 0, 6, 8, 0, 0., 2],
    }
    selected = st.selectbox("Cargar ejemplo ilustrativo", list(examples))
    values = examples[selected]
    with st.form("student_form"):
        a, b, c = st.columns(3)
        record = {}
        settings = [(17,70,1), (0.,200.,.1), (0.,200.,.1), (0,20,1), (0,26,1), (0,45,1), (0,26,1), (0.,20.,.1), (0,12,1)]
        helptext = ["Edad en años al ingresar.", "Escala portuguesa de ingreso, 0 a 200.", "Escala portuguesa de formación previa, 0 a 200.", "Asignaturas reconocidas o acreditadas en primer semestre.", "Total de asignaturas matriculadas en primer semestre.", "Total de intentos de evaluación; puede superar las asignaturas.", "Total aprobado en primer semestre.", "Promedio de asignaturas aprobadas; usa 0 cuando no hay aprobadas.", "Asignaturas sin intento de evaluación en primer semestre."]
        for i, feature in enumerate(FEATURES):
            with [a,b,c][i%3]:
                lo, hi, step = settings[i]
                record[feature] = st.number_input(FEATURE_LABELS[feature], min_value=lo, max_value=hi, value=values[i], step=step, key=f"{selected}_{i}", help=helptext[i])
        submitted = st.form_submit_button("Estimar trayectoria", type="primary", width="stretch")
    if submitted:
        errors = validate_record(record)
        if errors:
            for error in errors:
                st.error(error)
        else:
            result = predict_record(model, record)
            st.subheader("Resultado de la simulación")
            st.success(f"Resultado con mayor puntuación: {result['resultado']}")
            cols = st.columns(3)
            for col, (key, probability) in zip(cols, result["probabilidades"].items()):
                col.metric(LABELS[key], f"{probability:.1%}")
            chart = pd.DataFrame({"Resultado": [LABELS[k] for k in result["probabilidades"]], "Estimación": list(result["probabilidades"].values())}).set_index("Resultado")
            st.bar_chart(chart, horizontal=True, color="#167d8d")
            st.caption("Puntuaciones probabilísticas del modelo sin calibración adicional. No representan una certeza ni una causa del abandono.")
            st.download_button("Descargar resultado JSON", json.dumps({"entrada":record, **result}, ensure_ascii=False, indent=2), "simulacion.json", "application/json")
    st.warning("Uso académico. El modelo no ha sido validado en Colombia y no debe determinar sanciones, admisiones ni exclusiones. Una situación de dificultad requiere revisión humana y acompañamiento.")

elif page == "Resultados del modelo":
    st.subheader("Evaluación con datos reservados")
    t = metrics["test"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Exactitud", f"{t['accuracy']:.1%}")
    c2.metric("F1 macro", f"{t['f1_macro']:.3f}")
    c3.metric("Sensibilidad al abandono", f"{t['classification_report']['Dropout']['recall']:.1%}")
    c4.metric("Casos de prueba", metrics["n_test"])
    st.markdown(f"**Modelo seleccionado:** {metrics['modelo_seleccionado']}. Selección por F1 macro en validación cruzada de cinco particiones, usando únicamente entrenamiento.")
    st.dataframe(pd.DataFrame(metrics["comparacion_cv"])[["modelo","f1_macro_cv","desviacion_cv"]].rename(columns={"modelo":"Modelo","f1_macro_cv":"F1 macro CV","desviacion_cv":"Desviación CV"}), hide_index=True, width="stretch")
    a,b=st.columns(2)
    with a: st.image(str(ROOT/"figures/confusion_matrix.png"), caption="Matriz de confusión en prueba", width="stretch")
    with b: st.image(str(ROOT/"figures/class_metrics.png"), caption="Precisión, sensibilidad y F1 por clase", width="stretch")
    st.caption(f"Prueba reservada del 20 % ({metrics['n_test']} casos); entrenamiento del 80 % ({metrics['n_train']}). No se utilizó la prueba para elegir el modelo.")
    low, high=t['f1_macro_ic95_bootstrap']
    st.markdown(f"F1 macro: **{t['f1_macro']:.3f}**; IC bootstrap del 95 %: **[{low:.3f}, {high:.3f}]**. Intervalo condicional a este modelo y esta partición; no incluye incertidumbre por reentrenamiento.")
    st.markdown(f"Línea base de clase mayoritaria: exactitud **{metrics['baseline_test']['accuracy']:.1%}**, F1 macro **{metrics['baseline_test']['f1_macro']:.3f}**.")

elif page == "Datos y metodología":
    st.subheader("De los datos a la aplicación")
    st.markdown("**Fuente:** UCI Predict Students’ Dropout and Academic Success (Realinho et al., 2021). 4.424 registros y 36 predictores originales; se seleccionaron nueve por disponibilidad temporal y sencillez de uso.")
    st.markdown("**Objetivo:** estimar las tres trayectorias finales con variables de ingreso y de primer semestre. Se conservan los casos que siguen matriculados como una clase propia.")
    st.markdown("**Preparación:** revisión de faltantes, duplicados, tipos y coherencia. La copia descargada no contiene faltantes ni duplicados. Se conservan los valores extremos válidos; cero aprobadas y promedio cero tienen significado.")
    st.markdown("**Entrenamiento:** separación estratificada 80/20 y semilla 42; imputación de mediana dentro del pipeline como protección para faltantes futuros; escalado solo en regresión logística. Comparación de regresión logística, bosque aleatorio y gradient boosting.")
    st.markdown("**Evaluación:** F1 macro como criterio principal, exactitud, exactitud balanceada, métricas por clase y matriz de confusión. F1 macro asigna el mismo peso a cada clase (scikit-learn developers, s. f.-d).")
    st.image(str(ROOT/"figures/class_distribution.png"), caption="Distribución del conjunto completo para describir la muestra", width="stretch")
    st.markdown("**Límites:** datos de una institución portuguesa; evaluación retrospectiva aleatoria, sin validación temporal ni externa. Algunos abandonos pueden preceder el cierre del semestre: se requiere una cohorte con estudiantes activos y fechas verificadas antes de utilizar el sistema como alerta prospectiva. No se ha comprobado calibración ni equidad entre grupos. Una asociación predictiva no demuestra causalidad.")
    st.caption("Datos con licencia CC BY 4.0. Modelo de nueve variables: no utiliza sexo, nacionalidad ni datos del segundo semestre; ello no garantiza ausencia de sesgos.")

else:
    st.subheader("Referencias en APA 7")
    refs = json.loads((ROOT/"references.json").read_text(encoding="utf-8"))
    for ref in refs:
        st.markdown(ref["markdown"])
    st.caption("Notebook y archivos auxiliares documentan el experimento y permiten reproducirlo.")
