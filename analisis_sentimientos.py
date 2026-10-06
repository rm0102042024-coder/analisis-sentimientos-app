import streamlit as st
import pandas as pd
import io
import google.generativeai as genai
import re

# ── Configuración de la página ──────────────────────────────
st.set_page_config(
    page_title="Sentify | Análisis de Sentimiento",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Se agregó CSS específico para el título de la barra lateral
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0px; }
    .sub-title { color: #555555; font-size: 1rem; margin-bottom: 20px; }
    .sidebar-title { font-size: 1.8rem; font-weight: 800; color: #FF4B4B; margin-top: -10px; line-height: 1.1;}
    .sidebar-subtitle { font-size: 1rem; font-weight: 600; color: #888888; margin-bottom: 15px;}
    </style>
""", unsafe_allow_html=True)

# ── 1. Diccionarios de Reglas Académicas ────────────────────────
DICCIONARIO = {
    "excelente": 0.9, "deliciosa": 0.9, "riquísima": 0.9,
    "caliente": 0.7, "buena": 0.7,
    "rápido": 0.6, "limpio": 0.6,
    "frío": -0.5, "aceptable": -0.5,
    "lento": -0.6, "feo": -0.6,
    "malagana": -0.7, "sucio": -0.7, 
    "grosero": -0.9, "pésimo": -0.9
}

INTENSIFICADORES = {"muy": 1.5, "bastante": 1.5, "extremadamente": 2.0, "sumamente": 2.0}
NEGACIONES = {"no", "nunca", "jamás"}
SUSTANTIVOS_NEUTROS = {"comida", "servicio", "mesero", "postre", "ambiente", "hamburguesa", "pizza"}

# ── 2. Funciones Lógicas (Motor Matemático y API) ────────────────
def limpiar_texto_gemini(texto: str, api_key: str) -> str:
    """Usa Gemini para corregir ortografía y traducir al español si es necesario."""
    if not api_key:
        return texto # Falla de forma segura si no hay clave
    try:
        genai.configure(api_key=api_key)
        modelo = genai.GenerativeModel('gemini-3.8-flash')
        prompt = f"""
        Actúa como un corrector ortográfico. Toma el siguiente texto, tradúcelo al español si está en otro idioma, 
        y corrige cualquier error de ortografía o gramática. Devuelve ÚNICAMENTE el texto limpio, en minúsculas y sin puntuación extra:
        '{texto}'
        """
        respuesta = modelo.generate_content(prompt)
        return respuesta.text.strip().lower().replace(".", "").replace(",", "")
    except Exception as e:
        st.error(f"Detalle del error de Gemini: {e}")
        return texto.lower().replace(".", "").replace(",", "")

def evaluar_clausula(texto_clausula: str) -> float:
    """Aplica Regla 1 (Intensificación) y Regla 2 (Negación Sintáctica)"""
    palabras = texto_clausula.split()
    puntaje_clausula = 0.0
    multiplicador_actual = 1.0
    es_negacion = False
    
    for palabra in palabras:
        if palabra in NEGACIONES:
            es_negacion = True
        elif palabra in INTENSIFICADORES:
            multiplicador_actual = INTENSIFICADORES[palabra]
        elif palabra in DICCIONARIO:
            valor_base = DICCIONARIO[palabra]
            if es_negacion:
                valor_final = valor_base * -0.5
            else:
                valor_final = valor_base * multiplicador_actual
            puntaje_clausula += valor_final
            multiplicador_actual = 1.0
            es_negacion = False
        elif palabra in SUSTANTIVOS_NEUTROS:
            puntaje_clausula += 0.0 
            
    return puntaje_clausula

def predict_sentiment(text: str, api_key: str) -> dict:
    """Función principal que procesa el texto, aplica reglas y emite el veredicto"""
    if not text or not str(text).strip():
        return None

    # Preprocesamiento inteligente con Gemini
    texto_limpio = limpiar_texto_gemini(str(text), api_key)
    texto_evaluacion = texto_limpio.replace("mala gana", "malagana") # Unificar términos compuestos
    
    # Regla 3: Re-ponderación por Conjunción
    if " pero " in texto_evaluacion:
        partes = texto_evaluacion.split(" pero ")
        c1, c2 = partes[0], partes[1]
        puntaje_total = (evaluar_clausula(c1) * 0.4) + (evaluar_clausula(c2) * 1.0)
    elif " sin embargo " in texto_evaluacion:
        partes = texto_evaluacion.split(" sin embargo ")
        c1, c2 = partes[0], partes[1]
        puntaje_total = (evaluar_clausula(c1) * 0.4) + (evaluar_clausula(c2) * 1.0)
    else:
        puntaje_total = evaluar_clausula(texto_evaluacion)
        
    puntaje_total = round(puntaje_total, 2)
    
    # Regla 4: Matriz de Clasificación
    if puntaje_total >= 1.0:
        sentiment, emoji, status_type = "Recomendado ampliamente", "🤩", "success"
    elif 0.2 <= puntaje_total < 1.0:
        sentiment, emoji, status_type = "Experiencia Satisfactoria", "😊", "success"
    elif -0.2 < puntaje_total < 0.2:
        sentiment, emoji, status_type = "Opinión Indiferente/Neutra", "😐", "info"
    elif -1.0 < puntaje_total <= -0.2:
        sentiment, emoji, status_type = "Atención/Comida Deficiente", "😞", "error"
    else:
        sentiment, emoji, status_type = "Alerta/Mala Experiencia Crítica", "😡", "error"

    return {
        "text_original": str(text),
        "text_analyzed": texto_limpio,
        "sentiment": sentiment,
        "emoji": emoji,
        "status_type": status_type,
        "polarity": puntaje_total,
    }

# ── Estado de Sesión (Historial) ───────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Barra Lateral (Sidebar) ────────────────────────────────
with st.sidebar:
    # Se agregó el Título de la Aplicación en la barra lateral
    st.markdown('<div class="sidebar-title">Análisis de Sentimientos</div>', unsafe_allow_html=True)
    st.divider()
    
    st.header("⚙️ Configuración")
    api_key_input = st.text_input("Clave API de Gemini:", type="password", help="Obtén tu clave gratuita en Google AI Studio.")
    
    st.divider()
    st.markdown("### 📌 Guía de Clasificación")
    st.markdown("- **≥ +1.0:** Recomendado ampliamente")
    st.markdown("- **+0.2 a +0.99:** Satisfactoria")
    st.markdown("- **-0.2 a +0.19:** Neutra")
    st.markdown("- **-1.0 a -0.21:** Deficiente")
    st.markdown("- **≤ -1.0:** Experiencia Crítica")
    
    if st.button("🗑️ Limpiar Historial", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ── Encabezado Principal ───────────────────────────────────
st.markdown('<p class="main-title">🧠 Sentify — Análisis Basado en Reglas</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Evaluación léxica de polaridades con limpieza impulsada por Inteligencia Artificial.</p>', unsafe_allow_html=True)

if not api_key_input:
    st.warning("⚠️ Ingresa tu clave de API de Gemini en el menú lateral para activar la limpieza y traducción inteligente de texto.")

tab_analysis, tab_history, tab_file = st.tabs(["📝 Análisis de Texto", "📊 Historial", "📁 Analizar Archivo (CSV/TXT)"])

# ── Pestaña 1: Formulario y Resultados ──────────────────────
with tab_analysis:
    with st.form("sentiment_form", clear_on_submit=False):
        user_text = st.text_area("Texto a analizar:", placeholder="Escribe aquí tu frase...", height=130)
        submit_button = st.form_submit_button("Analizar Sentimiento 🚀", use_container_width=True)

    if submit_button and user_text.strip():
        with st.spinner("Limpiando y evaluando reglas matemáticas..."):
            result = predict_sentiment(user_text, api_key_input)
            st.session_state.history.append(result)

        st.divider()
        c_emoji, c_status, c_trans = st.columns([1, 3, 3])
        with c_emoji: st.markdown(f"# {result['emoji']}")
        with c_status:
            st.markdown(f"**Resultado Final:**")
            if result['status_type'] == "success": st.success(result['sentiment'])
            elif result['status_type'] == "error": st.error(result['sentiment'])
            else: st.info(result['sentiment'])
        with c_trans:
            st.caption("🤖 **Texto procesado por Gemini:**")
            st.info(f'"{result["text_analyzed"]}"')

        # Se eliminó la subjetividad (no es parte de la regla) y se ajustó la vista del puntaje
        st.metric("Puntaje Matemático Total", f"{result['polarity']}")

# ── Pestaña 2: Historial de la Sesión ──────────────────────
with tab_history:
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history[["text_original", "sentiment", "polarity"]], use_container_width=True)
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

        try:
            if file_extension == "txt":
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                text_list = [line.strip() for line in stringio.readlines() if line.strip()]
            elif file_extension == "csv":
                df_upload = pd.read_csv(uploaded_file)
                text_col = df_upload.select_dtypes(include=['object']).columns[0]
                text_list = df_upload[text_col].dropna().astype(str).tolist()
                st.info(f"Columna detectada para análisis: **{text_col}**")
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

        if text_list:
            if st.button(f"Analizar {len(text_list)} frases con Reglas 🚀", type="primary"):
                progress_bar = st.progress(0, text="Analizando...")
                results_file = []
                
                for i, phrase in enumerate(text_list):
                    res = predict_sentiment(phrase, api_key_input)
                    if res: results_file.append(res)
                    progress_bar.progress((i + 1) / len(text_list), text=f"Analizando... {i+1}/{len(text_list)}")
                
                progress_bar.empty()
                df_results = pd.DataFrame(results_file)
                
                st.success("¡Análisis completado!")
                st.dataframe(df_results[["text_original", "sentiment", "polarity"]], use_container_width=True)
                
                csv_download = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar resultados en CSV",
                    data=csv_download,
                    file_name="resultados_sentimiento_reglas.csv",
                    mime="text/csv",
                )