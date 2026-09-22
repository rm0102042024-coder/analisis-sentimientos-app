import streamlit as st
from textblob import TextBlob
from deep_translator import GoogleTranslator
import pandas as pd
import io

# ── Configuración de la página ──────────────────────────────
st.set_page_config(
    page_title="Sentify | Análisis de Sentimiento",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #555555;
        font-size: 1rem;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ── Funciones de Lógica de Negocio ─────────────────────────
@st.cache_data(show_spinner=False)
def translate_to_english(text: str) -> tuple[str, bool]:
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        is_translated = translated.strip().lower() != text.strip().lower()
        return translated, is_translated
    except Exception:
        return text, False

def predict_sentiment(text: str, auto_translate: bool = True) -> dict:
    if not text or not str(text).strip():
        return None

    analysis_text = str(text)
    was_translated = False

    if auto_translate:
        analysis_text, was_translated = translate_to_english(analysis_text)

    blob = TextBlob(analysis_text)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)

    if polarity > 0.1:
        sentiment, emoji, status_type = "Positivo", "😊", "success"
    elif polarity < -0.1:
        sentiment, emoji, status_type = "Negativo", "😞", "error"
    else:
        sentiment, emoji, status_type = "Neutral", "😐", "info"

    return {
        "text_original": str(text),
        "text_analyzed": analysis_text,
        "was_translated": was_translated,
        "sentiment": sentiment,
        "emoji": emoji,
        "status_type": status_type,
        "polarity": polarity,
        "subjectivity": subjectivity,
    }

# ── Estado de Sesión (Historial) ───────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Barra Lateral (Sidebar) ────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    auto_translate = st.checkbox(
        "Traducción Automática (Auto → EN)", 
        value=True,
        help="Detecta el idioma original y lo traduce al inglés para mayor precisión con TextBlob."
    )
    
    st.divider()
    st.markdown("### 📌 Guía de Métricas")
    st.markdown("**Polaridad:** Rango `[-1.0 a 1.0]`. Negativo a positivo.")
    st.markdown("**Subjetividad:** Rango `[0.0 a 1.0]`. Objetivo a subjetivo.")
    
    if st.button("🗑️ Limpiar Historial", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ── Encabezado Principal ───────────────────────────────────
st.markdown('<p class="main-title">🧠 Sentify — Análisis de Sentimiento</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Evalúa las emociones y el grado de subjetividad de cualquier texto en tiempo real.</p>', unsafe_allow_html=True)

tab_analysis, tab_history, tab_file = st.tabs(["📝 Análisis de Texto", "📊 Historial", "📁 Analizar Archivo (CSV/TXT)"])

# ── Pestaña 1: Formulario y Resultados ──────────────────────
with tab_analysis:
    with st.form("sentiment_form", clear_on_submit=False):
        user_text = st.text_area("Texto a analizar:", placeholder="Escribe aquí tu frase...", height=130)
        submit_button = st.form_submit_button("Analizar Sentimiento 🚀", use_container_width=True)

    if submit_button and user_text.strip():
        with st.spinner("Procesando análisis..."):
            result = predict_sentiment(user_text, auto_translate)
            st.session_state.history.append(result)

        st.divider()
        c_emoji, c_status, c_trans = st.columns([1, 3, 3])
        with c_emoji: st.markdown(f"# {result['emoji']}")
        with c_status:
            st.markdown(f"**Resultado:**")
            if result['status_type'] == "success": st.success(f"Sentimiento {result['sentiment']}")
            elif result['status_type'] == "error": st.error(f"Sentimiento {result['sentiment']}")
            else: st.info(f"Sentimiento {result['sentiment']}")
        with c_trans:
            if result['was_translated']:
                st.caption("🌐 **Texto traducido para el análisis:**")
                st.info(f'"{result["text_analyzed"]}"')

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Polaridad", f"{result['polarity']:.3f}")
            st.progress((result["polarity"] + 1) / 2, text="← Negativo | Positivo →")
        with m2:
            st.metric("Subjetividad", f"{result['subjectivity']:.3f}")
            st.progress(result["subjectivity"], text="← Objetivo | Subjetivo →")

# ── Pestaña 2: Historial de la Sesión ──────────────────────
with tab_history:
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history[["text_original", "sentiment", "polarity", "subjectivity"]], use_container_width=True)
        st.bar_chart(df_history['sentiment'].value_counts())
    else:
        st.info("Aún no has analizado ningún texto en esta sesión.")

# ── Pestaña 3: Carga de Archivos ───────────────────────────
with tab_file:
    st.markdown("Sube un archivo con varias frases para analizarlas en bloque.")
    uploaded_file = st.file_uploader("Formatos soportados: .TXT o .CSV", type=["txt", "csv"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        text_list = []

        # Procesar archivo según su tipo
        try:
            if file_extension == "txt":
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                text_list = [line.strip() for line in stringio.readlines() if line.strip()]
            elif file_extension == "csv":
                df_upload = pd.read_csv(uploaded_file)
                # Intenta usar la primera columna que contenga texto
                text_col = df_upload.select_dtypes(include=['object']).columns[0]
                text_list = df_upload[text_col].dropna().astype(str).tolist()
                st.info(f"Columna detectada para análisis: **{text_col}**")
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

        # Analizar en bloque
        if text_list:
            
            if st.button(f"Analizar {len(text_list)} frases 🚀", type="primary"):
                progress_bar = st.progress(0, text="Analizando...")
                results_file = []
                
                for i, phrase in enumerate(text_list):
                    res = predict_sentiment(phrase, auto_translate)
                    if res: results_file.append(res)
                    progress_bar.progress((i + 1) / len(text_list), text=f"Analizando... {i+1}/{len(text_list)}")
                
                progress_bar.empty()
                df_results = pd.DataFrame(results_file)
                
                st.success("¡Análisis completado!")
                st.dataframe(df_results[["text_original", "sentiment", "polarity"]], use_container_width=True)
                
                # Botón para descargar resultados
                csv_download = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar resultados en CSV",
                    data=csv_download,
                    file_name="resultados_sentimiento.csv",
                    mime="text/csv",
                )