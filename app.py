import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
import google.generativeai as genai
from deep_translator import GoogleTranslator
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Análisis Bursátil Profesional",
    page_icon="📈",
    layout="wide",
)

# Aplicar estilo CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    .stock-name {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .stock-info {
        font-size: 1.1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .summary-text {
        color: #0F172A;  /* Azul marino oscuro para el texto del resumen */
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .info-text {
        color: #0F172A;  /* Azul marino oscuro para la información actual */
        font-size: 1rem;
    }
    .plotly-graph {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.markdown("<h1 class='main-header'>Análisis Bursátil Profesional</h1>", unsafe_allow_html=True)

# Sidebar interactivo
with st.sidebar:
    st.header("Configuración del Análisis")
    
    ticker_symbol = st.text_input("Ingrese el Ticker Bursátil (Ej: AAPL)", "").upper()
    
    # Selector de período
    period_options = {
        "1 año": "1y",
        "3 años": "3y",
        "5 años": "5y",
        "Max": "max"
    }
    selected_period = st.selectbox("Seleccione el período de análisis", 
                                   list(period_options.keys()), 
                                   index=2)  # Default: 5 años
    
    # Índices de referencia para comparar
    st.subheader("Índices de Referencia")
    compare_spy = st.checkbox("S&P 500 (SPY)", value=True)
    compare_dji = st.checkbox("Dow Jones (DJI)", value=False)
    compare_qqq = st.checkbox("NASDAQ 100 (QQQ)", value=False)
    
    # Configuración de visualización
    st.subheader("Opciones de Visualización")
    show_volume = st.checkbox("Mostrar Volumen", value=True)
    use_log_scale = st.checkbox("Escala Logarítmica", value=False)
    show_moving_averages = st.checkbox("Mostrar Medias Móviles", value=True)
    
    # Expander para información adicional
    with st.expander("Más Información", expanded=False):
        st.markdown("""
        Esta aplicación analiza datos bursátiles y proporciona:
        - Resumen de la empresa generado por IA
        - Análisis de precios históricos
        - Comparación con índices de referencia
        - Métricas de rendimiento y riesgo
        - Visualizaciones interactivas
        
        Los datos se obtienen de Yahoo Finance y el resumen se genera usando la API Gemini de Google.
        """)

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.sidebar.error(f"Error al configurar Gemini: {e}")
    model = None

# Inicializar traductor
translator = GoogleTranslator(source='auto', target='es')

@st.cache_data(ttl=3600)  # Caché de datos con tiempo de vida de 1 hora
def get_stock_data(ticker, period):
    """Obtiene los datos históricos de un ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period=period)
        return info, hist
    except Exception as e:
        st.error(f"Error al obtener datos para {ticker}: {e}")
        return None, None

@st.cache_data(ttl=3600)
def get_benchmark_data(benchmark, period):
    """Obtiene los datos de un índice de referencia."""
    try:
        benchmark_data = yf.download(benchmark, period=period)
        return benchmark_data
    except Exception as e:
        st.error(f"Error al obtener datos para {benchmark}: {e}")
        return None

def generate_company_summary(company_name, sector, description, model):
    """Genera un resumen de la empresa usando Gemini."""
    if model and description and description != 'No disponible':
        prompt = f"""
        Genera un resumen conciso y profesional de la siguiente empresa:
        
        Nombre: {company_name}
        Sector: {sector}
        Descripción: {description}
        
        Incluye: principales actividades, mercados en los que opera, ventajas competitivas y aspectos financieros relevantes.
        El resumen debe tener entre 3-5 frases y ser informativo.
        """
        
        try:
            response = model.generate_content(prompt)
            if response.text:
                translated_summary = translator.translate(response.text)
                return translated_summary
            return None
        except Exception as e:
            st.error(f"Error con Gemini: {e}")
            return None
    return None

def translate_text(text):
    """Traduce texto al español."""
    if text and text != 'No disponible':
        try:
            # Para textos largos, los dividimos en partes para evitar límites de la API
            max_chunk_size = 4800  # Límite aproximado para GoogleTranslator
            if len(text) > max_chunk_size:
                chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                translated_chunks = [translator.translate(chunk) for chunk in chunks]
                return ' '.join(translated_chunks)
            else:
                return translator.translate(text)
        except Exception as e:
            st.error(f"Error al traducir texto: {e}")
    return text

def calculate_financial_metrics(data, price_column='Adj Close'):
    """Calcula métricas financieras importantes."""
    if data is None or data.empty or price_column not in data.columns:
        return None
    
    # Obtener precios
    prices = data[price_column]
    
    # Fechas para diferentes períodos
    today = prices.index[-1]
    one_month_ago = today - pd.DateOffset(months=1)
    six_months_ago = today - pd.DateOffset(months=6)
    one_year_ago = today - pd.DateOffset(years=1)
    three_years_ago = today - pd.DateOffset(years=3)
    five_years_ago = today - pd.DateOffset(years=5)
    
    # Precios para diferentes períodos
    price_today = prices.iloc[-1]
    price_1m_ago = prices.asof(one_month_ago) if one_month_ago >= prices.index[0] else None
    price_6m_ago = prices.asof(six_months_ago) if six_months_ago >= prices.index[0] else None
    price_1y_ago = prices.asof(one_year_ago) if one_year_ago >= prices.index[0] else None
    price_3y_ago = prices.asof(three_years_ago) if three_years_ago >= prices.index[0] else None
    price_5y_ago = prices.asof(five_years_ago) if five_years_ago >= prices.index[0] else None
    
    # Calcular retornos para diferentes períodos
    returns = {}
    
    # Retornos simples
    if price_1m_ago is not None and price_1m_ago > 0:
        returns['1m'] = (price_today / price_1m_ago - 1) * 100
    if price_6m_ago is not None and price_6m_ago > 0:
        returns['6m'] = (price_today / price_6m_ago - 1) * 100
    if price_1y_ago is not None and price_1y_ago > 0:
        returns['1y'] = (price_today / price_1y_ago - 1) * 100
    
    # Calcular CAGRs
    def calculate_cagr(start_price, end_price, years):
        if start_price is None or end_price is None or start_price <= 0 or years <= 0:
            return None
        return (pow(end_price / start_price, 1 / years) - 1) * 100
    
    returns['cagr_1y'] = calculate_cagr(price_1y_ago, price_today, 1)
    returns['cagr_3y'] = calculate_cagr(price_3y_ago, price_today, 3)
    returns['cagr_5y'] = calculate_cagr(price_5y_ago, price_today, 5)
    
    # Calcular volatilidad (anualizada)
    daily_returns = prices.pct_change().dropna()
    if not daily_returns.empty:
        returns['volatility'] = np.std(daily_returns) * math.sqrt(252) * 100
        
        # Sharpe Ratio (asumiendo una tasa libre de riesgo de 2%)
        risk_free_rate = 2.0  # 2% anual
        excess_return = daily_returns.mean() * 252 * 100 - risk_free_rate
        returns['sharpe_ratio'] = excess_return / returns['volatility'] if returns['volatility'] > 0 else None
        
        # Drawdown máximo
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1)
        returns['max_drawdown'] = drawdown.min() * 100
    
    return returns

def format_percentage(value):
    """Formatea un valor numérico como porcentaje con signo."""
    if value is None:
        return "N/A"
    return f"{'+' if value > 0 else ''}{value:.2f}%"

# --- APLICACIÓN PRINCIPAL ---
if ticker_symbol:
    # Obtener datos del ticker
    period = period_options[selected_period]
    company_info, historical_data = get_stock_data(ticker_symbol, period)
    
    if company_info and historical_data is not None and not historical_data.empty:
        # --- SECCIÓN DE INFORMACIÓN DE LA EMPRESA ---
        company_name = company_info.get('longName', ticker_symbol)
        sector = company_info.get('sector', 'No disponible')
        industry = company_info.get('industry', 'No disponible')
        raw_description = company_info.get('longBusinessSummary', 'No disponible')
        
        # Crear dos columnas para mostrar información
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"<h2 class='stock-name'>{company_name} ({ticker_symbol})</h2>", unsafe_allow_html=True)
            
            st.markdown(f"<p class='stock-info'><strong>Sector:</strong> {sector} | <strong>Industria:</strong> {industry}</p>", unsafe_allow_html=True)
            
            # Traducir y mostrar descripción original
            with st.spinner("Traduciendo descripción..."):
                translated_description = translate_text(raw_description)
                
            with st.expander("Descripción Original (Español)", expanded=False):
                st.markdown(f"<div class='info-text'>{translated_description}</div>", unsafe_allow_html=True)
            
            # Generar resumen con IA
            st.subheader("Resumen Ejecutivo")
            with st.spinner("Generando resumen..."):
                company_summary = generate_company_summary(company_name, sector, raw_description, model)
                if company_summary:
                    st.markdown(f"<div class='metric-card'><p class='summary-text'>{company_summary}</p></div>", unsafe_allow_html=True)
                else:
                    st.info("No se pudo generar el resumen de la empresa.")
        
        with col2:
            # Mostrar información financiera y bursátil clave
            current_price = historical_data['Close'].iloc[-1] if not historical_data.empty else None
            previous_close = historical_data['Close'].iloc[-2] if len(historical_data) > 1 else None
            
            price_change = ((current_price / previous_close) - 1) * 100 if current_price and previous_close else None
            price_color = "green" if price_change and price_change > 0 else "red"
            
            st.markdown("<h3>Información Bursátil</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Precio Actual</h4>
                <h2 class='info-text'>${current_price:.2f} <span style='color:{price_color};'>{format_percentage(price_change)}</span></h2>
            </div>
            """, unsafe_allow_html=True)
            
            market_cap = company_info.get('marketCap', None)
            if market_cap:
                if market_cap >= 1_000_000_000_000:  # Trillones
                    market_cap_str = f"${market_cap/1_000_000_000_000:.2f} T"
                elif market_cap >= 1_000_000_000:  # Billones
                    market_cap_str = f"${market_cap/1_000_000_000:.2f} B"
                elif market_cap >= 1_000_000:  # Millones
                    market_cap_str = f"${market_cap/1_000_000:.2f} M"
                else:
                    market_cap_str = f"${market_cap:,.0f}"
                
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Capitalización de Mercado</h4>
                    <h3 class='info-text'>{market_cap_str}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Mostrar algunos ratios clave si están disponibles
            pe_ratio = company_info.get('trailingPE', None)
            dividend_yield = company_info.get('dividendYield', None) * 100 if company_info.get('dividendYield', None) else None
            
            if pe_ratio or dividend_yield:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                if pe_ratio:
                    st.markdown(f"<p class='info-text'><strong>P/E Ratio:</strong> {pe_ratio:.2f}</p>", unsafe_allow_html=True)
                if dividend_yield:
                    st.markdown(f"<p class='info-text'><strong>Dividend Yield:</strong> {dividend_yield:.2f}%</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        # --- SECCIÓN DE ANÁLISIS TÉCNICO ---
        st.markdown("<h2 class='section-header'>Análisis Técnico</h2>", unsafe_allow_html=True)
        
        # Obtener datos de índices para comparar
        benchmark_data = {}
        if compare_spy:
            benchmark_data['SPY'] = get_benchmark_data('SPY', period)
        if compare_dji:
            benchmark_data['DJI'] = get_benchmark_data('^DJI', period)
        if compare_qqq:
            benchmark_data['QQQ'] = get_benchmark_data('QQQ', period)
        
        # Preparar datos para gráficos comparativos
        price_column = 'Adj Close' if 'Adj Close' in historical_data.columns else 'Close'
        
        # 1. Gráfico de precios y volumen interactivo con Plotly
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           row_heights=[0.7, 0.3] if show_volume else [1, 0])
        
        # Añadir línea de precio
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data[price_column],
                name=ticker_symbol,
                line=dict(color='#1E3A8A', width=2),
            ),
            row=1, col=1
        )
        
        # Añadir medias móviles si están activadas
        if show_moving_averages:
            # Media móvil de 50 días
            ma_50 = historical_data[price_column].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=ma_50,
                    name='MA 50',
                    line=dict(color='#FF6B6B', width=1, dash='dash'),
                ),
                row=1, col=1
            )
            
            # Media móvil de 200 días
            ma_200 = historical_data[price_column].rolling(window=200).mean()
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=ma_200,
                    name='MA 200',
                    line=dict(color='#4BC0C0', width=1, dash='dash'),
                ),
                row=1, col=1
            )
        
        # Añadir volumen si está activado
        if show_volume:
            colors = ['#27AE60' if row['Close'] >= row['Open'] else '#E74C3C' 
                     for _, row in historical_data.iterrows()]
            
            fig.add_trace(
                go.Bar(
                    x=historical_data.index,
                    y=historical_data['Volume'],
                    marker_color=colors,
                    name='Volumen',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Configurar el diseño
        fig.update_layout(
            title=f"Evolución de Precio de {company_name} ({ticker_symbol})",
            xaxis_title="Fecha",
            yaxis_title="Precio",
            height=600,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        
        if use_log_scale:
            fig.update_yaxes(type="log", row=1, col=1)
        
        # Mostrar el gráfico
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # 2. Gráfico comparativo con índices de referencia
        # Corregir el error de "The truth value of a DataFrame is ambiguous"
        has_benchmarks = False
        for benchmark in benchmark_data.values():
            if benchmark is not None and not benchmark.empty:
                has_benchmarks = True
                break
                
        if has_benchmarks:
            st.subheader("Rendimiento Comparativo (Base 100)")
            
            # Preparar datos para comparación
            # Busca esta sección en el código, alrededor de la línea 450-470
# Preparar datos para comparación
        comparison_data = pd.DataFrame()

        # Normalizar precios (Base 100)
        first_valid_date = historical_data.index[0]
        comparison_data[ticker_symbol] = historical_data[price_column] / historical_data[price_column].iloc[0] * 100

        # Añadir índices de referencia
        for name, data in benchmark_data.items():
            if data is not None and not data.empty:
                # Encontrar el precio más cercano a la fecha inicial del ticker principal
                benchmark_price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                benchmark_price = data[benchmark_price_col].asof(first_valid_date.tz_localize(data.index.tz))
                if benchmark_price is not None and not benchmark_price.empty and benchmark_price.iloc[0] > 0:
                    comparison_data[name] = data[benchmark_price_col] / benchmark_price.iloc[0] * 100
            
            # Crear el gráfico de comparación
            fig_comparison = px.line(
                comparison_data,
                x=comparison_data.index,
                y=comparison_data.columns,
                title=f"Rendimiento comparativo (Base 100) - {ticker_symbol} vs Índices de Referencia",
                labels={'value': 'Rendimiento (Base 100)', 'variable': 'Activo'},
                template='plotly_white',
                color_discrete_sequence=['#1E3A8A', '#FF6B6B', '#4BC0C0', '#FFD166'],
            )
            
            fig_comparison.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # --- SECCIÓN DE MÉTRICAS FINANCIERAS ---
        st.markdown("<h2 class='section-header'>Métricas de Rendimiento y Riesgo</h2>", unsafe_allow_html=True)
        
        metrics = calculate_financial_metrics(historical_data, price_column)
        
        if metrics:
            col1, col2, col3 = st.columns(3)
            
            # Rendimientos por período
            with col1:
                st.markdown("<h3>Rendimientos</h3>", unsafe_allow_html=True)
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                
                # Mostrar rendimientos para diferentes períodos
                if 'cagr_1y' in metrics:
                    st.markdown(f"<p class='info-text'><strong>CAGR 1 año:</strong> {format_percentage(metrics['cagr_1y'])}</p>", unsafe_allow_html=True)
                if 'cagr_3y' in metrics:
                    st.markdown(f"<p class='info-text'><strong>CAGR 3 años:</strong> {format_percentage(metrics['cagr_3y'])}</p>", unsafe_allow_html=True)
                if 'cagr_5y' in metrics:
                    st.markdown(f"<p class='info-text'><strong>CAGR 5 años:</strong> {format_percentage(metrics['cagr_5y'])}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Rendimientos a corto plazo
            with col2:
                st.markdown("<h3>Rendimientos Recientes</h3>", unsafe_allow_html=True)
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                
                if '1m' in metrics:
                    st.markdown(f"<p class='info-text'><strong>1 mes:</strong> {format_percentage(metrics['1m'])}</p>", unsafe_allow_html=True)
                if '6m' in metrics:
                    st.markdown(f"<p class='info-text'><strong>6 meses:</strong> {format_percentage(metrics['6m'])}</p>", unsafe_allow_html=True)
                if '1y' in metrics:
                    st.markdown(f"<p class='info-text'><strong>1 año:</strong> {format_percentage(metrics['1y'])}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Métricas de riesgo
            with col3:
                st.markdown("<h3>Indicadores de Riesgo</h3>", unsafe_allow_html=True)
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                
                if 'volatility' in metrics:
                    st.markdown(f"<p class='info-text'><strong>Volatilidad (anual):</strong> {format_percentage(metrics['volatility'])}</p>", unsafe_allow_html=True)
                if 'max_drawdown' in metrics:
                    st.markdown(f"<p class='info-text'><strong>Máximo Drawdown:</strong> {format_percentage(metrics['max_drawdown'])}</p>", unsafe_allow_html=True)
                if 'sharpe_ratio' in metrics:
                    sharpe_display = f"{metrics['sharpe_ratio']:.2f}" if metrics['sharpe_ratio'] is not None else "N/A"
                    st.markdown(f"<p class='info-text'><strong>Ratio de Sharpe:</strong> {sharpe_display}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Explicación de métricas
            with st.expander("Explicación de Métricas"):
                st.markdown("""
                ### Glosario de Métricas
                
                #### Rendimientos
                - **CAGR (Compound Annual Growth Rate)**: Tasa de crecimiento anual compuesto, representa el rendimiento anualizado de la inversión.
                - **Rendimientos Recientes**: Variación porcentual del precio en diferentes períodos de tiempo.
                
                #### Riesgo
                - **Volatilidad**: Desviación estándar anualizada de los rendimientos diarios. Una mayor volatilidad indica mayor riesgo.
                - **Máximo Drawdown**: La mayor caída porcentual desde un máximo hasta un mínimo subsiguiente. Indica el peor escenario histórico.
                - **Ratio de Sharpe**: Rendimiento ajustado por riesgo. Mide el exceso de rendimiento por unidad de riesgo adicional. Mayor es mejor.
                """)
        
        # --- SECCIÓN DE DISTRIBUCIÓN DE RENDIMIENTOS ---
        st.markdown("<h2 class='section-header'>Distribución de Rendimientos Diarios</h2>", unsafe_allow_html=True)
        
        # Calcular retornos diarios
        if price_column in historical_data.columns:
            daily_returns = historical_data[price_column].pct_change().dropna() * 100
            
            # Crear histograma con Plotly
            fig_hist = px.histogram(
                daily_returns,
                nbins=50,
                title="Distribución de Rendimientos Diarios (%)",
                labels={'value': 'Rendimiento Diario (%)', 'count': 'Frecuencia'},
                template='plotly_white',
                color_discrete_sequence=['#1E3A8A'],
            )
            
            # Añadir línea vertical para la media
            mean_return = daily_returns.mean()
            fig_hist.add_vline(x=mean_return, line_width=2, line_dash="dash", line_color="red",
                              annotation_text=f"Media: {mean_return:.2f}%",
                              annotation_position="top right")
            
            fig_hist.update_layout(
                showlegend=False,
                height=400,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
    else:
        st.error(f"No se encontraron datos para {ticker_symbol}. Verifica que el ticker sea correcto.")
else:
    # Pantalla inicial cuando no se ha ingresado un ticker
    st.markdown("""
    ### 👋 ¡Bienvenido al Análisis Bursátil Profesional!
    
    Para comenzar, ingresa un símbolo bursátil (ticker) en el panel lateral izquierdo.
    
    #### Características principales:
    - Análisis de precios históricos con gráficos interactivos
    - Comparación con índices de mercado (S&P 500, Dow Jones, NASDAQ)
    - Cálculo de rendimientos y métricas de riesgo
    - Resumen de la empresa generado por IA
    
    #### Ejemplos de tickers populares:
    - AAPL (Apple)
    - MSFT (Microsoft)
    - AMZN (Amazon)
    - GOOGL (Alphabet/Google)
    - TSLA (Tesla)
    - JPM (JPMorgan Chase)
    
    *Ingresa un ticker y haz clic fuera del campo de texto para iniciar el análisis.*
    """)