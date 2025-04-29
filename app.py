import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
import google.generativeai as genai
from deep_translator import GoogleTranslator
import os

st.set_page_config(page_title="Análisis Financiero Avanzado", page_icon="📈", layout="wide")
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stApp { background-color: #f4f6f8; }
    .reportview-container { background: #f4f6f8; }
    .sidebar .st-header { padding-bottom: 1rem; border-bottom: 1px solid #ccc; }
    .st-metric-label { font-size: 1rem; font-weight: bold; color: #336699; }
    .st-metric-value { font-size: 1.5rem; color: #2e4053; }
    .st-metric-delta { font-size: 1rem; color: green; }
    .st-metric-delta.negative { color: red; }
    .section-header { font-size: 1.6rem; font-weight: bold; color: #1c395e; margin-top: 1.5rem; }
    .ai-analysis { background-color: #e9ecef; padding: 1rem; border-radius: 5px; margin-top: 1rem; }
    .macro-data-guidance { background-color: #f0f8ea; padding: 1rem; border-radius: 5px; margin-top: 1rem; color: #2e7d32; }
    .full-description { margin-top: 1rem; padding: 1rem; border: 1px solid #ccc; border-radius: 5px; background-color: #fff; }
    .popular-action-button { margin-right: 0.5rem; }
    .performance-formula { background-color: #f9f9f9; padding: 0.75rem; border: 1px solid #ddd; border-radius: 3px; margin-top: 1rem; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

# --- FUNCIONES UTILITARIAS ---
def calculate_cagr(initial_value, final_value, years):
    """Calcula la Tasa de Crecimiento Anual Compuesto."""
    if initial_value <= 0 or years <= 0:
        return 0
    return (final_value / initial_value) ** (1/years) - 1

def calculate_annualized_volatility(returns):
    """Calcula la volatilidad anualizada a partir de los rendimientos diarios."""
    daily_std = np.std(returns)
    return daily_std * np.sqrt(252)

def format_percentage(value):
    """Formatea un número como una cadena de porcentaje."""
    return f"{value * 100:.2f}%"

def get_ai_summary(text, max_length=150):
    """Genera un resumen conciso del texto utilizando la IA de Gemini."""
    if not api_key or api_key == "TU_API_KEY_AQUI":
        return "La funcionalidad de resumen AI no está disponible. Por favor, configura la API key de Gemini."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')

        prompt = f"""
            Por favor, proporciona un resumen conciso en español del siguiente texto,
            limitándolo a un máximo de {max_length} palabras:

            {text}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error al generar el resumen AI: {e}")
        return "No se pudo generar un resumen."

def get_ai_recommendation(company_name, sector, current_price, volatility_str, cagr_1yr, cagr_3yr, cagr_5yr):
    """Genera una recomendación breve basada en IA."""
    if not api_key or api_key == "TU_API_KEY_AQUI":
        return "La funcionalidad de análisis AI no está disponible. Por favor, configura la API key de Gemini."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')

        prompt = f"""
            Genera una recomendación concisa para un inversor sobre la acción de {company_name},
            que opera en el sector de {sector}. El precio actual es de ${{current_price}},
            la volatilidad anualizada es de {volatility_str}, y las tasas de crecimiento anual compuesto
            (CAGR) son: 1 año = {cagr_1yr}, 3 años = {cagr_3yr}, 5 años = {cagr_5yr}.
            Considera estos factores para dar una perspectiva general (alcista, neutral o bajista)
            y un breve razonamiento de no más de dos frases. El idioma de la respuesta debe ser español.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error al generar la recomendación AI: {e}")
        return None
def get_macroeconomic_guidance(sector, industry):
    """Proporciona una guía sobre dónde buscar datos macroeconómicos."""
    guidance = f"""
        Para un análisis más profundo de {industry} dentro del sector de {sector},
        considera explorar las siguientes fuentes de datos macroeconómicos:
        \n\n
        - **Informes Gubernamentales:** Busca informes de instituciones gubernamentales relevantes en México
          (ej., INEGI, Banco de México) y a nivel internacional (ej., FMI, Banco Mundial) sobre el sector y la economía en general.
        - **Indicadores Económicos:** Presta atención a indicadores como el PIB, la inflación, las tasas de interés,
          el desempleo y la confianza del consumidor. Estos pueden influir significativamente en el rendimiento de las acciones.
        - **Noticias Financieras y Análisis de Mercado:** Sigue fuentes de noticias financieras confiables (ej., El Financiero, Expansión, Reuters, Bloomberg)
          y los análisis de mercado de bancos de inversión y firmas de análisis.
        - **Datos del Sector Específico:** Investiga informes y datos de asociaciones industriales y empresas de investigación de mercado
          que se especialicen en el sector de {sector} y la industria de {industry}.
        - **Calendario Económico:** Mantente al tanto de los próximos eventos económicos y publicaciones de datos que podrían afectar al mercado.
        \n
        Analizar cómo estos factores macroeconómicos podrían impactar las perspectivas de crecimiento y la rentabilidad de {industry}
        puede proporcionar una visión más completa de la inversión en esta acción.
    """
    return guidance
def main():
    # Título principal
    st.title("📈 Análisis Financiero Avanzado de Acciones")
    st.markdown("---")
    # Sidebar para la configuración
    with st.sidebar:
        st.header("⚙️ Configuración del Análisis")
        ticker = st.text_input("Símbolo de la Acción (ej: AAPL)", "").upper()
        translate_desc = st.checkbox("Traducir descripción al inglés", value=False)
        st.markdown("---")
        st.markdown("""
            ### Instrucciones:
            1. Ingrese el símbolo de la acción.
            2. Opcional: Active la traducción de la descripción.
            """)
    # Contenido principal si se ingresa un ticker
    if ticker:
        try:
            # Obtener datos de la acción
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info.get('longName'):
                st.error("❌ Símbolo de acción no válido o datos no disponibles.")
                return
            # --- SECCIÓN DE INFORMACIÓN DE LA EMPRESA ---
            st.header(f"🏢 Información de la Empresa: {info.get('longName', ticker)}")
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.subheader("Detalles Generales")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industria:** {info.get('industry', 'N/A')}")
                st.write(f"**Símbolo Bursátil:** {ticker}")
                st.write(f"**Bolsa:** {info.get('exchange', 'N/A')}")
            with col_info2:
                st.subheader("Precio Actual")
                st.metric(
                    label="Precio",
                    value=f"${info.get('currentPrice', 0):.2f}",
                    delta=f"{info.get('regularMarketChangePercent', 0):.2f}%"
                )
            # Descripción de la empresa
            st.subheader("📝 Descripción de la Empresa")
            description = info.get('longBusinessSummary', 'No disponible')
            summary = get_ai_summary(description)
            st.write(f"**Resumen:** {summary}")
            show_full_description = st.checkbox("Ver descripción completa")
            if show_full_description:
                st.markdown(f"<div class='full-description'><h4>Descripción Completa:</h4> {description}</div>", unsafe_allow_html=True)
            if translate_desc:
                try:
                    translated = GoogleTranslator(source='auto', target='en').translate(description)
                    st.subheader("📝 Descripción (Inglés)")
                    st.write(translated)
                except Exception as e:
                    st.warning("No se pudo traducir la descripción.")
            # --- SECCIÓN DE ANÁLISIS HISTÓRICO DEL PRECIO ---
            st.header("📈 Análisis Histórico del Precio")
            hist = stock.history(period="10y") # Extendemos el historial para tener más opciones
            if len(hist) < 252:
                st.error("❌ No hay suficientes datos históricos para el análisis.")
                return
            # Obtener datos de los índices
            indices = ["SPY", "DJI", "QQQ"]
            index_data = yf.download(indices, start=hist.index.min(), end=hist.index.max())['Close']
            index_data_normalized = index_data / index_data.iloc[0]
            stock_data_normalized = hist['Close'] / hist['Close'].iloc[0]
            # Crear la gráfica comparativa
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Scatter(x=hist.index, y=stock_data_normalized, mode='lines', name=info.get("longName", ticker)))
            for index in indices:
                fig_comparison.add_trace(go.Scatter(x=index_data_normalized.index, y=index_data_normalized[index], mode='lines', name=index))
            fig_comparison.update_layout(
                title=f'Comparativa del Precio Normalizado de {info.get("longName", ticker)} vs. Índices',
                xaxis_title="Fecha",
                yaxis_title="Precio Normalizado (Base 1)",
                template='plotly_white'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            # --- SECCIÓN DE MÉTRICAS DE RENDIMIENTO Y RIESGO ---
            st.header("📊 Métricas de Rendimiento y Riesgo (Últimos 5 Años)")
            hist['Returns'] = hist['Close'].pct_change().dropna()
            col_metrics1, col_metrics2 = st.columns(2)
            with col_metrics1:
                st.subheader("Rendimiento Anualizado (CAGR)")
                current_price = hist['Close'].iloc[-1]
                for years in [1, 3, 5]:
                    if len(hist) >= years * 252:
                        initial_price = hist['Close'].iloc[-(years * 252)]
                        cagr_value = calculate_cagr(initial_price, current_price, years)
                        st.metric(label=f"CAGR {years} Año(s)", value=format_percentage(cagr_value))
                    else:
                        st.metric(label=f"CAGR {years} Año(s)", value="N/A")
            with col_metrics2:
                st.subheader("Riesgo")
                volatility = calculate_annualized_volatility(hist['Returns'])
                st.metric(label="Volatilidad Anualizada", value=format_percentage(volatility))
                st.info("""
                    La volatilidad anualizada mide la dispersión de los rendimientos de la acción a lo largo de un año.
                    Un valor más alto indica un mayor riesgo.
                """)
            # --- SECCIÓN DE RENDIMIENTO HISTÓRICO (SELECCIÓN DE PERIODO) ---
            st.subheader("🗓️ Rendimiento Histórico (Selección de Periodo)")
            performance_years = st.selectbox("Seleccionar periodo para calcular el rendimiento:", [3, 5, 10])
            if performance_years:
                num_years = int(performance_years)
                if len(hist) >= num_years * 252:
                    initial_price_perf = hist['Close'].iloc[-(num_years * 252)]
                    final_price_perf = hist['Close'].iloc[-1]
                    total_return = (final_price_perf / initial_price_perf) - 1
                    st.metric(label=f"Rendimiento Total (Últimos {num_years} Años)", value=format_percentage(total_return))
                else:
                    st.warning(f"No hay suficientes datos para calcular el rendimiento de los últimos {num_years} años.")
            st.markdown("""
                <div class='performance-formula'>
                    **Fórmula para el Rendimiento Total:**
                    $$\\text{Rendimiento Total} = \\frac{\\text{Precio Final} - \\text{Precio Inicial}}{\\text{Precio Inicial}}$$
                </div>
            """, unsafe_allow_html=True)
            # --- SECCIÓN DE RECOMENDACIÓN BASADA EN IA ---
            st.header("🤖 Recomendación Basada en IA")
            if api_key and api_key != "TU_API_KEY_AQUI":
                with st.spinner("Generando recomendación AI..."):
                    cagr_1yr_str = format_percentage(calculate_cagr(hist['Close'].iloc[-252], current_price, 1)) if len(hist) >= 252 else "N/A"
                    cagr_3yr_str = format_percentage(calculate_cagr(hist['Close'].iloc[-252*3], current_price, 3)) if len(hist) >= 252*3 else "N/A"
                    cagr_5yr_str = format_percentage(calculate_cagr(hist['Close'].iloc[-252*5], current_price, 5)) if len(hist) >= 252*5 else "N/A"
                    ai_recommendation = get_ai_recommendation(
                        info.get('longName', ticker),
                        info.get('sector', 'N/A'),
                        info.get('currentPrice', 0),
                        format_percentage(volatility),
                        cagr_1yr_str,
                        cagr_3yr_str,
                        cagr_5yr_str
                    )
                    if ai_recommendation:
                        st.markdown(f"<div class='ai-analysis'><h4>Análisis AI:</h4> {ai_recommendation}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("No se pudo generar una recomendación AI en este momento.")
            else:
                    st.warning("La API key de Google Gemini no está configurada. La recomendación AI no está disponible.")
            # --- GUÍA DE DATOS MACROECONÓMICOS ---
            st.header("🌍 Guía para Datos Macroeconómicos")
            macro_guidance = get_macroeconomic_guidance(info.get('sector', 'N/A'), info.get('industry', 'N/A'))
            st.markdown(f"<div class='macro-data-guidance'><h4>Búsqueda de Datos Macroeconómicos:</h4> {macro_guidance}</div>", unsafe_allow_html=True)
        except yf.YFinanceError as e:
            st.error(f"❌ Error al obtener datos de Yahoo Finance: {e}")
        except Exception as e:
            st.error(f"❌ Ocurrió un error inesperado: {e}")
    else:
        st.info("👈 Por favor, ingrese un símbolo de acción en el panel lateral para comenzar el análisis avanzado.")
    # --- SECCIÓN DE ACCIONES POPULARES ---
    st.subheader("🔥 Acciones Populares")
    popular_actions = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "BRK.A", "JPM", "V", "JNJ"]
    cols = st.columns(len(popular_actions))
    for i, action in enumerate(popular_actions):
        with cols[i]:
            if st.button(action, key=f"popular_{action}", help=f"Analizar {action}"):
                st.rerun()
                st.session_state.ticker = action
                st.query_params["ticker"] = action
    if "ticker" in st.session_state and st.session_state.ticker:
        main() # Volver a ejecutar la función principal con el ticker seleccionado
    # --- SECCIÓN DE CONFIGURACIÓN DE API KEY ---       
