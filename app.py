import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import google.generativeai as genai
from deep_translator import GoogleTranslator

# ----------------- CONFIGURACIÓN GEMINI -----------------
api_key = "AIzaSyCPyb9KQcsRe87k_T9WJmLTHtIt340pHHw"  # Considera usar st.secrets para manejar API keys de forma segura
genai.configure(api_key=api_key)
modelo = genai.GenerativeModel("gemini-1.5-flash")

# ----------------- CONFIG DE LA PÁGINA -----------------
st.set_page_config(page_title="Análisis Financiero con Gemini", layout="wide")
st.title("📊 Análisis Financiero de Acciones + Gemini")

# ----------------- INPUT DEL USUARIO -----------------
symbol = st.text_input("🔤 Ingresa el símbolo bursátil (por ejemplo: AAPL, MSFT, TSLA):")

if symbol:
    try:
        # ----------------- DATOS DE LA EMPRESA -----------------
        st.subheader("🏢 Descripción de la Empresa")
        ticker = yf.Ticker(symbol)
        info = ticker.info

        sector = info.get("sector", "No disponible")
        descripcion = info.get("longBusinessSummary", "No se encontró descripción.")
        
        try:
            descripcion_traducida = GoogleTranslator(source='auto', target='es').translate(descripcion)
        except Exception as e:
            st.warning(f"No se pudo traducir la descripción: {e}")
            descripcion_traducida = descripcion

        st.markdown(f"**Industria:** {sector}")
        st.markdown(descripcion_traducida)

        # ----------------- ANÁLISIS DE ÍNDICES -----------------
        st.subheader("📈 Análisis Financiero: Índices Comparables")
        peers = ["SPY", "QQQ", "^DJI", symbol.upper()]
        precios_cierre = pd.DataFrame()

        for ticker in peers:
            try:
                data = yf.download(ticker, period="5y")
                if not data.empty:
                    if 'Adj Close' in data.columns:
                        precios_cierre[ticker] = data['Adj Close']
                    elif 'Close' in data.columns:
                        precios_cierre[ticker] = data['Close']
                    else:
                        st.warning(f"No se encontró 'Adj Close' ni 'Close' para {ticker}. Se omitirá.")
                else:
                    st.warning(f"No se encontraron datos para {ticker}. Se omitirá.")
            except Exception as e:
                st.warning(f"Error al descargar datos para {ticker}: {e}")

        if not precios_cierre.empty:
            precios_cierre = precios_cierre.dropna()
            
            if not precios_cierre.empty:
                rendimientos = np.log(precios_cierre / precios_cierre.shift(1))
                tasa_libre_riesgo = 0.02
                tasa_diaria = tasa_libre_riesgo / 252

                estadisticas = pd.DataFrame(index=precios_cierre.columns)
                estadisticas["Máximo"] = precios_cierre.max()
                estadisticas["Mínimo"] = precios_cierre.min()
                estadisticas["Desviación Estándar Anual"] = rendimientos.std() * np.sqrt(252)
                
                mean_rendimientos = rendimientos.mean()
                if not (mean_rendimientos == 0).any():
                    estadisticas["Coef. de Variación"] = estadisticas["Desviación Estándar Anual"] / mean_rendimientos
                else:
                    estadisticas["Coef. de Variación"] = np.nan
                    st.warning("Algunos rendimientos medios son cero, no se puede calcular el coeficiente de variación")

                st.markdown("Este cálculo considera el precio al inicio y al final del periodo para determinar el rendimiento anualizado.")

                # ----------------- CÁLCULO DE RENDIMIENTOS ANUALIZADOS -----------------
                st.subheader("📊 Rendimientos Anualizados")
                precios_anuales = precios_cierre.copy()
                años = [1, 3, 5]
                rendimiento_anual = {}

                for año in años:
                    dias = 252 * año
                    if len(precios_anuales) >= dias:
                        precios_final = precios_anuales.iloc[-1]
                        precios_inicio = precios_anuales.iloc[-dias]
                        rendimiento = ((precios_final / precios_inicio) ** (1 / año)) - 1
                        rendimiento_anual[f"{año} años"] = rendimiento
                    else:
                        st.warning(f"No hay suficientes datos para calcular el rendimiento de {año} año(s)")

                if rendimiento_anual:
                    tabla_rendimiento = pd.DataFrame(rendimiento_anual)
                    st.dataframe(tabla_rendimiento.style.format("{:.2%}"), use_container_width=True)
                else:
                    st.warning("No se pudo calcular ningún rendimiento anualizado por falta de datos")

                # -------------------📈 GRÁFICO COMPARATIVO -------------------
                st.subheader("📉 Evolución Histórica Comparativa (5 años)")
                precios_normalizados = precios_cierre / precios_cierre.iloc[0] * 100

                fig = go.Figure()
                for ticker in precios_normalizados.columns:
                    fig.add_trace(go.Scatter(
                        x=precios_normalizados.index,
                        y=precios_normalizados[ticker],
                        mode='lines',
                        name=ticker
                    ))

                fig.update_layout(
                    title="Comparativa de Rendimiento (Base 100)",
                    xaxis_title="Fecha",
                    yaxis_title="Índice Normalizado",
                    template="plotly_dark"
                )

                st.plotly_chart(fig, use_container_width=True)

                # ----------------- ANÁLISIS FINANCIERO AVANZADO -----------------
                st.subheader("📋 Análisis Financiero Avanzado")
                instrucciones = """
Actúa como un analista financiero de élite, con experiencia en análisis técnico y fundamental.
Utiliza los siguientes datos estadísticos de los índices SPY, QQQ y DJI para hacer un análisis detallado que incluya:
- 📌 Resumen ejecutivo
- 🌍 Contexto macroeconómico
- 📈 Análisis técnico
- ⚠️ Evaluación y gestión de riesgos
- 🔮 Propuesta de escenarios

No hagas predicciones directas ni recomendaciones de inversión, mantén la objetividad.
Datos a analizar:
"""
                prompt_analisis = instrucciones + str(estadisticas)
                try:
                    respuesta2 = modelo.generate_content(prompt_analisis)
                    st.info(respuesta2.text)
                except Exception as e:
                    st.error(f"Error al generar el análisis con Gemini: {e}")
            else:
                st.error("No hay suficientes datos válidos para continuar con el análisis")
        else:
            st.error("No se pudieron obtener datos de los índices comparables")

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")