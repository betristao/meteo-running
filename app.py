"""
Weather Decision Support System — Portugal Running
Análise de histórico meteorológico (2016-2026) para escolha de datas de eventos de running.
Fonte: Open-Meteo Historical Weather API (ERA5).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, date
import calendar
import concurrent.futures
from fpdf import FPDF
import io

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

CITIES = {
    # Norte
    "Viana do Castelo": (41.6932, -8.8329),
    "Braga":            (41.5518, -8.4229),
    "Vila Real":        (41.3006, -7.7441),
    "Bragança":         (41.8058, -6.7572),
    "Porto":            (41.1579, -8.6291),
    # Centro
    "Aveiro":           (40.6443, -8.6455),
    "Viseu":            (40.6566, -7.9125),
    "Guarda":           (40.5373, -7.2658),
    "Coimbra":          (40.2033, -8.4103),
    "Castelo Branco":   (39.8222, -7.4919),
    "Leiria":           (39.7436, -8.8071),
    "Santarém":         (39.2333, -8.6833),
    # Sul e Ilhas
    "Lisboa":           (38.7223, -9.1393),
    "Portalegre":       (39.2938, -7.4312),
    "Setúbal":          (38.5244, -8.8882),
    "Évora":            (38.5714, -7.9083),
    "Beja":             (38.0151, -7.8632),
    "Faro":             (37.0194, -7.9322),
    "Funchal":          (32.6669, -16.9241),
    "Ponta Delgada":    (37.7483, -25.6666),
}

MONTH_NAMES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}

WEEKDAY_NAMES_PT = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

# ──────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_data(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily historical weather from Open-Meteo, including sunrise/sunset."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,apparent_temperature_max,apparent_temperature_min,apparent_temperature_mean,precipitation_sum,wind_speed_10m_max,sunrise,sunset",
        "timezone": "Europe/Lisbon",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["daily"]
    df = pd.DataFrame({
        "date":           pd.to_datetime(data["time"]),
        "temp_max":       data["temperature_2m_max"],
        "temp_min":       data["temperature_2m_min"],
        "temp_avg":       data["temperature_2m_mean"],
        "app_temp_max":   data["apparent_temperature_max"],
        "app_temp_min":   data["apparent_temperature_min"],
        "app_temp_avg":   data["apparent_temperature_mean"],
        "precipitation":  data["precipitation_sum"],
        "wind_max":       data["wind_speed_10m_max"],
    })
    # Parse sunrise/sunset if available
    if "sunrise" in data and data["sunrise"]:
        df["sunrise"] = pd.to_datetime(data["sunrise"])
        df["sunset"]  = pd.to_datetime(data["sunset"])
        df["daylight_hours"] = (df["sunset"] - df["sunrise"]).dt.total_seconds() / 3600
    else:
        df["sunrise"] = pd.NaT
        df["sunset"]  = pd.NaT
        df["daylight_hours"] = np.nan
    df["month"]    = df["date"].dt.month
    df["day"]      = df["date"].dt.day
    df["year"]     = df["date"].dt.year
    df["weekday"]  = df["date"].dt.weekday  # 0=Mon … 6=Sun
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hourly_specific_day(lat: float, lon: float, month: int, day: int, years: list) -> pd.DataFrame:
    """Fetch hourly historical weather for a specific day across multiple years."""
    def _fetch_single_day(year):
        try:
            date_str = pd.to_datetime(f"{year}-{month:02d}-{day:02d}").strftime("%Y-%m-%d")
        except ValueError:
            return None # e.g., Feb 29 on non-leap year
            
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m,wind_direction_10m,shortwave_radiation",
            "timezone": "Europe/Lisbon",
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if "hourly" in data:
                    return data["hourly"]
        except Exception:
            pass
        return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(15, len(years))) as executor:
        futures = {executor.submit(_fetch_single_day, y): y for y in years}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                df_hr = pd.DataFrame(res)
                df_hr["time"] = pd.to_datetime(df_hr["time"])
                df_hr["hour"] = df_hr["time"].dt.hour
                df_hr["year"] = df_hr["time"].dt.year
                results.append(df_hr)
                
    if results:
        df_all = pd.concat(results, ignore_index=True)
        # Add running score for the hour using a simplified formula for visualization
        return df_all
    else:
        return pd.DataFrame()



# ──────────────────────────────────────────────
# RUNNING SCORE
# ──────────────────────────────────────────────

def compute_running_score(row) -> float:
    """
    Score 0-100 para provas de asfalto (5km a Maratona).
    Baseado em modelos de termorregulação desportiva.
    """
    score = 100.0

    # Rain penalty
    precip = row["precipitation"] if pd.notna(row["precipitation"]) else 0
    if precip > 15:
        score -= 50
    elif precip > 5:
        score -= 25
    elif precip > 1:
        score -= 10

    # Wind penalty
    wind = row["wind_max"] if pd.notna(row["wind_max"]) else 0
    if wind > 40:
        score -= 40
    elif wind > 25:
        score -= 20
    elif wind > 15:
        score -= 5

    # Apparent Temperature penalty (Sensação Térmica - inclui humidade/Heat Index)
    # Optimum range for racing: 7°C to 15°C feel
    app_temp = row["app_temp_avg"] if pd.notna(row["app_temp_avg"]) else 12
    if 5 <= app_temp <= 15:
        score -= 0
    elif 15 < app_temp <= 20:
        score -= 5
    elif 20 < app_temp <= 25:
        score -= 15
    elif app_temp > 25:
        score -= 35
    elif 1 <= app_temp < 5:
        score -= 5
    else:
        score -= 15

    return max(0.0, score)

def compute_hourly_score(row) -> float:
    """Variante horária do running score"""
    score = 100.0
    if row["precipitation"] > 2: score -= 50
    elif row["precipitation"] > 0.5: score -= 25
    elif row["precipitation"] > 0: score -= 10
    
    wind = row["wind_speed_10m"]
    if wind > 30: score -= 40
    elif wind > 20: score -= 20
    elif wind > 10: score -= 5
    
    app_temp = row["apparent_temperature"]
    if 5 <= app_temp <= 15: score -= 0
    elif 15 < app_temp <= 20: score -= 5
    elif 20 < app_temp <= 25: score -= 15
    elif app_temp > 25: score -= 35
    elif 1 <= app_temp < 5: score -= 5
    else: score -= 15
    return max(0.0, min(100.0, score))


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["running_score"] = df.apply(compute_running_score, axis=1)
    return df


# ──────────────────────────────────────────────
# KPI HELPERS
# ──────────────────────────────────────────────

def kpi_card(label: str, value: str, subtitle: str = "", delta_color: str = "normal"):
    """Render a styled KPI metric."""
    st.metric(label=label, value=value, delta=subtitle, delta_color=delta_color)


# ──────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────

def build_trend_chart(df: pd.DataFrame):
    """Dual-axis monthly trend: temperature + precipitation."""
    monthly = df.groupby(df["date"].dt.to_period("M")).agg(
        temp_avg=("temp_avg", "mean"),
        precipitation=("precipitation", "sum"),
    ).reset_index()
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=monthly["date"], y=monthly["temp_avg"],
            name="Temperatura Média (°C)",
            line=dict(color="#00d4aa", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.08)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=monthly["date"], y=monthly["precipitation"],
            name="Precipitação (mm)",
            marker_color="rgba(99,160,255,0.45)",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="°C", secondary_y=False, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="mm", secondary_y=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")

    return fig


def build_risk_heatmap(df: pd.DataFrame, month: int):
    """Day-of-month × Year heatmap coloured by average Running Score."""
    subset = df[df["month"] == month].copy()
    if subset.empty:
        return None

    pivot = subset.pivot_table(index="day", columns="year", values="running_score", aggfunc="mean")
    pivot = pivot.sort_index(ascending=True)

    fig = px.imshow(
        pivot.values,
        x=[str(y) for y in pivot.columns],
        y=[str(d) for d in pivot.index],
        color_continuous_scale=["#d32f2f", "#ff9800", "#fdd835", "#66bb6a", "#00c853"],
        zmin=0, zmax=100,
        aspect="auto",
        labels=dict(x="Ano", y="Dia", color="Score"),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(title="Score", tickvals=[0, 25, 50, 75, 100]),
    )
    # Ensure day 1 is at the top, and years are treated as discrete categories
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category")
    return fig


# ──────────────────────────────────────────────
# TOP WEEKENDS
# ──────────────────────────────────────────────

def find_top_weekends(df: pd.DataFrame, month: int, top_n: int = 3) -> pd.DataFrame:
    """Find best weekends (Sat+Sun) for a given month across all years."""
    subset = df[(df["month"] == month) & (df["weekday"].isin([5, 6]))].copy()
    if subset.empty:
        return pd.DataFrame()

    # Assign each Saturday+Sunday pair to the same "weekend_id"
    subset = subset.sort_values("date")
    subset["iso_week"] = subset["date"].dt.isocalendar().week.astype(int)
    subset["weekend_id"] = subset["year"].astype(str) + "-W" + subset["iso_week"].astype(str)

    weekends = subset.groupby("weekend_id").agg(
        data_inicio=("date", "min"),
        data_fim=("date", "max"),
        score_medio=("running_score", "mean"),
        precip_total=("precipitation", "sum"),
        vento_max=("wind_max", "max"),
        temp_media=("temp_avg", "mean"),
    ).reset_index()

    weekends = weekends.sort_values("score_medio", ascending=False).head(top_n)

    result = weekends[["data_inicio", "data_fim", "score_medio", "precip_total", "vento_max", "temp_media"]].copy()
    result.columns = ["Início", "Fim", "Score Médio", "Precip. Total (mm)", "Vento Máx (km/h)", "Temp. Média (°C)"]
    result["Início"] = result["Início"].dt.strftime("%d/%m/%Y")
    result["Fim"]    = result["Fim"].dt.strftime("%d/%m/%Y")
    result["Score Médio"]       = result["Score Médio"].round(1)
    result["Precip. Total (mm)"] = result["Precip. Total (mm)"].round(1)
    result["Vento Máx (km/h)"]  = result["Vento Máx (km/h)"].round(1)
    result["Temp. Média (°C)"]  = result["Temp. Média (°C)"].round(1)

    return result.reset_index(drop=True)


def find_best_weekends_statistical(df: pd.DataFrame, month: int, top_n: int = 3) -> pd.DataFrame:
    """
    Statistical approach: find all actual Sat-Sun weekends for the given month
    across all years, group by week of the month, average scores, and rank.
    """
    subset = df[(df["month"] == month) & (df["weekday"].isin([5, 6]))].copy()
    if subset.empty:
        return pd.DataFrame()

    # Categorize by week of the month purely by day number:
    # Week 1: 1-7, Week 2: 8-14, Week 3: 15-21, Week 4: 22-28, Week 5: 29+
    subset["semana_mes"] = ((subset["day"] - 1) // 7) + 1

    # For each week, compute the average stats across all years
    weekly = subset.groupby("semana_mes").agg(
        score_medio=("running_score", "mean"),
        prob_chuva=("precipitation", lambda x: round((x > 1).mean() * 100, 1)),
        vento_medio=("wind_max", "mean"),
        temp_media=("temp_avg", "mean"),
    ).reset_index()

    weekly = weekly.sort_values("score_medio", ascending=False).head(top_n)

    period_labels = {
        1: "1º Fim-de-semana (Dias 1-7)",
        2: "2º Fim-de-semana (Dias 8-14)",
        3: "3º Fim-de-semana (Dias 15-21)",
        4: "4º Fim-de-semana (Dias 22-28)",
        5: "5º Fim-de-semana (Dias 29-31)"
    }

    rows = []
    for _, r in weekly.iterrows():
        semana = int(r['semana_mes'])
        label = period_labels.get(semana, f"{semana}º Fim-de-semana")
        rows.append({
            "Período do Mês": label,
            "Score Medio": round(r["score_medio"], 1),
            "Prob. Chuva (%)": round(r["prob_chuva"], 1),
            "Vento Medio (km/h)": round(r["vento_medio"], 1),
            "Temp. Media (C)": round(r["temp_media"], 1),
        })

    return pd.DataFrame(rows).reset_index(drop=True)


# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    /* Global */
    .block-container { padding-top: 1.5rem; }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(26,31,46,0.9) 0%, rgba(14,17,23,0.9) 100%);
        border: 1px solid rgba(0,212,170,0.15);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: rgba(250,250,250,0.6) !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #00d4aa !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: rgba(250,250,250,0.85);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(0,212,170,0.2);
    }

    /* Top weekends table */
    .top-weekend-card {
        background: linear-gradient(135deg, rgba(26,31,46,0.95) 0%, rgba(14,17,23,0.95) 100%);
        border: 1px solid rgba(0,212,170,0.12);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #141926 100%);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        color: rgba(250,250,250,0.7) !important;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: rgba(250,250,250,0.35);
        font-size: 0.75rem;
        margin-top: 3rem;
        padding: 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.05);
    }

    /* Hide default Streamlit footer & hamburger */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .score-high   { background: rgba(0,200,83,0.2); color: #00c853; }
    .score-medium { background: rgba(253,216,53,0.2); color: #fdd835; }
    .score-low    { background: rgba(211,47,47,0.2); color: #d32f2f; }
</style>
"""


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Meteo Running Portugal",
        page_icon="🏃",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.markdown("## 🏃 Meteo Running")
        st.markdown('<p style="color:rgba(250,250,250,0.5); font-size:0.85rem; margin-top:-0.5rem;">Decisão baseada em dados meteorológicos</p>', unsafe_allow_html=True)
        st.divider()

        sorted_cities = sorted(list(CITIES.keys()))
        default_idx = sorted_cities.index("Lisboa") if "Lisboa" in sorted_cities else 0
        city = st.selectbox("Cidade", sorted_cities, index=default_idx)

        year_range = st.slider(
            "Intervalo de Anos",
            min_value=2016, max_value=2024,
            value=(2016, 2024),
        )

        all_months = list(MONTH_NAMES_PT.values())
        
        select_all_months = st.checkbox("Selecionar Todos os Meses", value=True)
        
        if select_all_months:
            selected_month_names = all_months
        else:
            selected_month_names = st.multiselect(
                "Escolha os meses:",
                options=all_months,
                default=[],
            )
            
        # Convert names back to numbers
        name_to_num = {v: k for k, v in MONTH_NAMES_PT.items()}
        selected_months = [name_to_num[m] for m in selected_month_names]

        st.divider()
        st.markdown(
            '<p style="color:rgba(250,250,250,0.3); font-size:0.72rem;">'
            'Powered by Open-Meteo ERA5<br>© 2026 Meteo Running</p>',
            unsafe_allow_html=True,
        )

    # ── Header ───────────────────────────────
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.markdown(f"# 🏃 Meteo Running — {city}")
        st.markdown(
            f'<p style="color:rgba(250,250,250,0.5); margin-top:-1rem;">'
            f'Análise meteorológica {year_range[0]}–{year_range[1]} · '
            f'{len(selected_months)} {"meses" if len(selected_months) != 1 else "mês"} seleccionado(s)</p>',
            unsafe_allow_html=True,
        )

    if not selected_months:
        st.warning("Seleccione pelo menos um mês para ver a análise.")
        return

    # ── Fetch data ───────────────────────────
    lat, lon = CITIES[city]
    start_date = f"{year_range[0]}-01-01"
    # Dynamic end date: use today's date but cap to avoid requesting future data
    from datetime import date as dt_date
    today = dt_date.today()
    end_year = min(year_range[1], today.year)
    end_date = today.strftime("%Y-%m-%d") if end_year == today.year else f"{end_year}-12-31"

    with st.spinner("A carregar dados meteorológicos…"):
        try:
            df = fetch_weather_data(lat, lon, start_date, end_date)
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            return

    df = add_scores(df)

    # Filter by selected months
    df_filtered = df[df["month"].isin(selected_months)].copy()

    if df_filtered.empty:
        st.warning("Sem dados disponíveis para os filtros seleccionados.")
        return

    # ── TABS DE NAVEGAÇÃO ────────────────────
    tab_dash, tab_best, tab_specific, tab_compare, tab_data = st.tabs([
        "📊 Dashboard Geral", 
        "🏆 Melhores Datas", 
        "🔍 Análise Específica", 
        "⚖️ Comparador", 
        "💾 Dados"
    ])

    with tab_dash:
        # ── KPI Cards ────────────────────────────
        st.markdown('<div class="section-header">📊 Indicadores-Chave</div>', unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)

        precip_prob = (df_filtered["precipitation"] > 1).mean() * 100
        avg_wind = df_filtered["wind_max"].mean()
        ideal_temp_pct = ((df_filtered["temp_avg"] >= 8) & (df_filtered["temp_avg"] <= 16)).mean() * 100
        avg_score = df_filtered["running_score"].mean()

        with k1:
            kpi_card("Prob. Precipitação", f"{precip_prob:.0f}%", "dias com > 1mm", "inverse")
        with k2:
            kpi_card("Vento Médio Máx", f"{avg_wind:.1f} km/h", "média diária")
        with k3:
            kpi_card("Temp. Ideal (8-16°C)", f"{ideal_temp_pct:.0f}%", "dos dias")
        with k4:
            score_label = "Excelente" if avg_score >= 75 else "Bom" if avg_score >= 50 else "Moderado"
            kpi_card("Running Score Médio", f"{avg_score:.0f}/100", score_label)

        # ── Trend Chart ──────────────────────────
        st.markdown('<div class="section-header">📈 Tendência Mensal — Temperatura & Precipitação</div>', unsafe_allow_html=True)
        trend_fig = build_trend_chart(df_filtered)
        st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})

        # ── Risk Heatmap ─────────────────────────
        st.markdown('<div class="section-header">🗓️ Heatmap de Risco por Mês</div>', unsafe_allow_html=True)

        # Show heatmaps in a tab layout for each selected month
        if len(selected_months) == 1:
            month_num = selected_months[0]
            fig = build_risk_heatmap(df_filtered, month_num)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info(f"Sem dados para {MONTH_NAMES_PT[month_num]}.")
        else:
            tab_labels = [MONTH_NAMES_PT[m] for m in sorted(selected_months)]
            tabs_heat = st.tabs(tab_labels)
            for t_heat, month_num in zip(tabs_heat, sorted(selected_months)):
                with t_heat:
                    fig = build_risk_heatmap(df_filtered, month_num)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.info(f"Sem dados para {MONTH_NAMES_PT[month_num]}.")

    with tab_best:
        # ── Top Weekends ─────────────────────────
        st.markdown('<div class="section-header">🏆 Top 3 Melhores Fins-de-semana (Estatístico)</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:rgba(250,250,250,0.45); font-size:0.82rem; margin-top:-0.5rem;">'
            'Média do Running Score para cada par Sábado-Domingo ao longo de todos os anos seleccionados.</p>',
            unsafe_allow_html=True,
        )

        if len(selected_months) == 1:
            month_num = selected_months[0]
            top_df = find_best_weekends_statistical(df_filtered, month_num)
            if not top_df.empty:
                st.dataframe(
                    top_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score Medio": st.column_config.ProgressColumn(
                            "Score Medio", min_value=0, max_value=100, format="%.0f",
                        ),
                    },
                )
                
                # Highlight absolute best day
                best_day_df = df_filtered[df_filtered["month"] == month_num].groupby("day").agg(score_medio=("running_score", "mean")).reset_index()
                if not best_day_df.empty:
                    best_day_row = best_day_df.loc[best_day_df["score_medio"].idxmax()]
                    st.markdown(f'''
                    <div style="background-color: rgba(69, 219, 142, 0.1); border-left: 4px solid #45DB8E; padding: 10px 15px; margin-top: 15px; border-radius: 4px;">
                        <strong>🥇 Melhor Dia Absoluto:</strong> Históricamente em {MONTH_NAMES_PT[month_num]}, o dia com melhores condições é o <strong>Dia {int(best_day_row["day"])}</strong> (Score Médio: {best_day_row["score_medio"]:.1f}/100).
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info(f"Sem dados de fins-de-semana para {MONTH_NAMES_PT[month_num]}.")
        else:
            tab_labels2 = [MONTH_NAMES_PT[m] for m in sorted(selected_months)]
            tabs_we = st.tabs(tab_labels2)
            for t_we, month_num in zip(tabs_we, sorted(selected_months)):
                with t_we:
                    top_df = find_best_weekends_statistical(df_filtered, month_num)
                    if not top_df.empty:
                        st.dataframe(
                            top_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Score Medio": st.column_config.ProgressColumn(
                                    "Score Medio", min_value=0, max_value=100, format="%.0f",
                                ),
                            },
                        )
                        
                        # Highlight absolute best day
                        best_day_df = df_filtered[df_filtered["month"] == month_num].groupby("day").agg(score_medio=("running_score", "mean")).reset_index()
                        if not best_day_df.empty:
                            best_day_row = best_day_df.loc[best_day_df["score_medio"].idxmax()]
                            st.markdown(f'''
                            <div style="background-color: rgba(69, 219, 142, 0.1); border-left: 4px solid #45DB8E; padding: 10px 15px; margin-top: 15px; border-radius: 4px;">
                                <strong>🥇 Melhor Dia Absoluto:</strong> Históricamente em {MONTH_NAMES_PT[month_num]}, o dia com melhores condições é o <strong>Dia {int(best_day_row["day"])}</strong> (Score Médio: {best_day_row["score_medio"]:.1f}/100).
                            </div>
                            ''', unsafe_allow_html=True)
                    else:
                        st.info(f"Sem dados de fins-de-semana para {MONTH_NAMES_PT[month_num]}.")

        # ── Historical detail per month ──────────
        st.markdown('<br><hr>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📅 Probabilidade Histórica por Dia</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:rgba(250,250,250,0.45); font-size:0.82rem; margin-top:-0.5rem;">'
            'Probabilidade estatística de chuva, temperatura média e score para cada dia do mês.</p>',
            unsafe_allow_html=True,
        )

        month_for_detail = st.selectbox(
            "Mês para análise detalhada",
            [MONTH_NAMES_PT[m] for m in sorted(selected_months)],
            index=0,
            key="detail_month",
        )
        detail_month_num = name_to_num[month_for_detail]

        detail_subset = df_filtered[df_filtered["month"] == detail_month_num]
        if not detail_subset.empty:
            daily_stats = detail_subset.groupby("day").agg(
                anos_analisados=("year", "nunique"),
                prob_chuva=("precipitation", lambda x: round((x > 1).mean() * 100, 1)),
                precip_media=("precipitation", "mean"),
                temp_media=("temp_avg", "mean"),
                vento_medio=("wind_max", "mean"),
                score_medio=("running_score", "mean"),
            ).reset_index()

            daily_stats.columns = [
                "Dia", "Anos Analisados", "Prob. Chuva (%)",
                "Precip. Média (mm)", "Temp. Média (°C)",
                "Vento Médio (km/h)", "Score Médio",
            ]
            daily_stats["Precip. Média (mm)"] = daily_stats["Precip. Média (mm)"].round(1)
            daily_stats["Temp. Média (°C)"]   = daily_stats["Temp. Média (°C)"].round(1)
            daily_stats["Vento Médio (km/h)"] = daily_stats["Vento Médio (km/h)"].round(1)
            daily_stats["Score Médio"]        = daily_stats["Score Médio"].round(1)

            st.dataframe(
                daily_stats,
                use_container_width=True,
                hide_index=True,
                height=400,
                column_config={
                    "Score Médio": st.column_config.ProgressColumn(
                        "Score Médio", min_value=0, max_value=100, format="%.0f",
                    ),
                    "Prob. Chuva (%)": st.column_config.ProgressColumn(
                        "Prob. Chuva (%)", min_value=0, max_value=100, format="%.0f",
                    ),
                },
            )

    with tab_specific:
        # ── Specific Day Report ──────────────────
        st.markdown('<div class="section-header">🔍 Relatório Específico de Prova</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#b2bec3; font-size:0.95rem;">'
            'Insira a data prevista para a prova e obtenha um relatório detalhado com base no histórico de todos os anos disponíveis (ignorando o filtro de meses lateral).</p>',
            unsafe_allow_html=True,
        )

        col_day, col_month = st.columns(2)
        with col_month:
            race_month_name = st.selectbox(
                "Mês da Prova",
                list(MONTH_NAMES_PT.values()),
                index=list(MONTH_NAMES_PT.values()).index(month_for_detail),
                key="race_month",
            )
        with col_day:
            race_month_num = name_to_num[race_month_name]
            # Calculate max days for chosen month (using 2024 to account for leap years)
            start_date_month = pd.to_datetime(f"2024-{race_month_num:02}-01")
            days_in_month = start_date_month.days_in_month
            race_day = st.number_input("Dia da Prova", min_value=1, max_value=days_in_month, value=15, step=1)

        # Use the full dataframe `df` filtered only by the selected year range and specific day/month
        race_data = df[
            (df["month"] == race_month_num) & 
            (df["day"] == race_day) & 
            (df["year"] >= year_range[0]) & 
            (df["year"] <= year_range[1])
        ]

        if not race_data.empty:
            years_count = race_data["year"].nunique()
            prob_rain = (race_data["precipitation"] > 1).mean() * 100
            avg_temp = race_data["temp_avg"].mean()
            max_temp_avg = race_data["temp_max"].mean()
            avg_wind = race_data["wind_max"].mean()
            avg_score = race_data["running_score"].mean()

            # Sunrise/Sunset info
            avg_sunrise = ""
            avg_sunset = ""
            avg_daylight = ""
            if race_data["sunrise"].notna().any():
                mean_sr_seconds = race_data["sunrise"].dropna().apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second).mean()
                sr_h, sr_m = int(mean_sr_seconds // 3600), int((mean_sr_seconds % 3600) // 60)
                avg_sunrise = f"{sr_h:02d}:{sr_m:02d}"
                mean_ss_seconds = race_data["sunset"].dropna().apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second).mean()
                ss_h, ss_m = int(mean_ss_seconds // 3600), int((mean_ss_seconds % 3600) // 60)
                avg_sunset = f"{ss_h:02d}:{ss_m:02d}"
                avg_daylight = f"{race_data['daylight_hours'].mean():.1f}h"

            tab_over, tab_sim, tab_anomalies, tab_hist = st.tabs(["📊 Visão Geral", "🤖 Simulações Pro", "🚨 Anomalias", "🗓️ Histórico Bruto"])
            with tab_over:
                # ── SEMÁFORO DE RISCO GLOBAL ────────────
                risk_score = 0
                risk_factors = []
                # Temperature risk
                if avg_temp > 28 or avg_temp < 2:
                    risk_score += 3
                    risk_factors.append("Temperatura extrema")
                elif avg_temp > 22 or avg_temp < 5:
                    risk_score += 2
                    risk_factors.append("Temperatura desfavorável")
                elif avg_temp > 18 or avg_temp < 8:
                    risk_score += 1
                # Rain risk
                if prob_rain > 50:
                    risk_score += 3
                    risk_factors.append("Probabilidade de chuva muito alta")
                elif prob_rain > 30:
                    risk_score += 2
                    risk_factors.append("Probabilidade de chuva moderada")
                elif prob_rain > 15:
                    risk_score += 1
                # Wind risk
                if avg_wind > 30:
                    risk_score += 3
                    risk_factors.append("Vento forte")
                elif avg_wind > 20:
                    risk_score += 2
                    risk_factors.append("Vento moderado")
                elif avg_wind > 12:
                    risk_score += 1

                if risk_score <= 2:
                    sem_color, sem_icon, sem_label, sem_desc = "#00c853", "🟢", "SEGURA", "Data aprovada. Condições históricas excelentes para a realização da prova."
                elif risk_score <= 5:
                    sem_color, sem_icon, sem_label, sem_desc = "#ff9800", "🟡", "PRECAUÇÃO", f"Data viável com precauções. Fatores de atenção: {', '.join(risk_factors) if risk_factors else 'condições marginais'}."
                else:
                    sem_color, sem_icon, sem_label, sem_desc = "#d32f2f", "🔴", "CRÍTICA", f"Data de alto risco. Fatores críticos: {', '.join(risk_factors)}. Considerar data alternativa."

                st.markdown(f'''
                <div style="background: linear-gradient(135deg, {sem_color}22, {sem_color}08); 
                            border: 2px solid {sem_color}; border-radius: 12px; 
                            padding: 20px 25px; margin-bottom: 20px; text-align: center;">
                    <div style="font-size: 2.5rem;">{sem_icon}</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: {sem_color}; margin: 5px 0;">DATA {sem_label}</div>
                    <div style="font-size: 0.95rem; color: #dfe6e9;">{sem_desc}</div>
                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.4); margin-top: 8px;">Índice de Risco: {risk_score}/9 · Score Climatérico: {avg_score:.0f}/100 · Baseado em {years_count} anos de dados ERA5 (ECMWF)</div>
                </div>
                ''', unsafe_allow_html=True)

                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Anos de Histórico", f"{years_count}")
                rc2.metric("Probabilidade Chuva >1mm", f"{prob_rain:.1f}%")
                rc3.metric("Temp. Média Esperada", f"{avg_temp:.1f} °C")
                rc4.metric("Score Histórico", f"{avg_score:.0f}/100")

                # Sunrise/Sunset row
                if avg_sunrise:
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("🌅 Nascer do Sol", avg_sunrise)
                    sc2.metric("🌇 Pôr do Sol", avg_sunset)
                    sc3.metric("☀️ Horas de Luz", avg_daylight)

                st.markdown("#### 🏃 Recomendações Técnicas para a Organização:")
            
                recommends = []
                if prob_rain > 30:
                    recommends.append("🌧 **Risco de Chuva Moderado/Alto:** Probabilidade considerável de chuva. Planear zonas de refúgio, reforço de impermeáveis para o staff técnico e sinalização de piso potencialmente escorregadio.")
                elif prob_rain > 15:
                    recommends.append("🌦 **Risco Baixo de Chuva:** Possibilidade de precipitação ligeira. Manter plano de contingência para chuva activo por precaução.")
                
                if max_temp_avg > 25 or avg_temp > 22:
                    recommends.append("🔥 **Calor:** As temperaturas podem ser exigentes para os atletas. Reforçar os postos de abastecimento de água (idealmente a cada 2.5km) e prever assistência médica reforçada com banhos de esponja ou túneis de água.")
                elif avg_temp < 7:
                    recommends.append("🧊 **Frio Exigente:** Temperaturas médias muito baixas. No regulamento recomendar vestuário adequado aos atletas (ex: manguitos, luvas) e garantir mantas térmicas na zona de meta.")
                
                if avg_wind > 20:
                    recommends.append("💨 **Vento Forte:** Previsão histórica de vento considerável. Reavaliar a fixação de pórticos insufláveis e montagem de estruturas altas (ex: tendas e bandeirolas Sponsor), pois o vento pode superar as recomendações de segurança.")

                # Sunrise-based recommendations
                if avg_sunrise and sr_h >= 7 and sr_m >= 30:
                    recommends.append(f"🌅 **Nascer do Sol Tardio ({avg_sunrise}):** O sol nasce relativamente tarde. Se a prova tem partida às 8h00, considerar iluminação artificial na zona de partida e coletes reflectores para os primeiros km.")
                
                if avg_score >= 80:
                    recommends.append("⭐ **Condições Excelentes:** A data apresenta estatisticamente excelentes condições meteorológicas para a prática de running e possibilita a quebra de recordes pessoais (PRs).")
                elif avg_score >= 70:
                    recommends.append("✅ **Boas Condições:** O histórico sugere boas condições para a organização da prova e conforto dos atletas.")
                
                if not recommends:
                    recommends.append("✅ **Condições Normais:** A data apresenta historicamente condições normais, sem grandes riscos que requeiram planeamento de contingência excepcional.")

                for rec in recommends:
                    st.info(rec)

                # ── Historical Extreme Alerts ────────────
                st.markdown("#### ⚠️ Piores Cenários Registados (Extremos Históricos):")
            
                worst_rain = race_data.loc[race_data["precipitation"].idxmax()]
                worst_wind = race_data.loc[race_data["wind_max"].idxmax()]
                worst_score = race_data.loc[race_data["running_score"].idxmin()]
                hottest = race_data.loc[race_data["temp_max"].idxmax()]
                coldest = race_data.loc[race_data["temp_min"].idxmin()]

                extreme_cols = st.columns(3)
                with extreme_cols[0]:
                    st.error(f"🌧 **Máx. Precipitação:** {worst_rain['precipitation']:.1f} mm em {worst_rain['date'].strftime('%d/%m/%Y')}")
                    st.error(f"💨 **Máx. Vento:** {worst_wind['wind_max']:.1f} km/h em {worst_wind['date'].strftime('%d/%m/%Y')}")
                with extreme_cols[1]:
                    st.warning(f"🔥 **Dia Mais Quente:** {hottest['temp_max']:.1f}°C em {hottest['date'].strftime('%d/%m/%Y')}")
                    st.warning(f"🧊 **Dia Mais Frio:** {coldest['temp_min']:.1f}°C em {coldest['date'].strftime('%d/%m/%Y')}")
                with extreme_cols[2]:
                    st.error(f"📉 **Pior Score:** {worst_score['running_score']:.0f}/100 em {worst_score['date'].strftime('%d/%m/%Y')}")
                    best_score_row = race_data.loc[race_data["running_score"].idxmax()]
                    st.success(f"📈 **Melhor Score:** {best_score_row['running_score']:.0f}/100 em {best_score_row['date'].strftime('%d/%m/%Y')}")

                # ── Executive Summary ────────────
                st.markdown("#### 📋 Resumo Executivo (Dossier PDF / Copiar):")
            
                risk_level = "Baixo" if prob_rain < 20 else ("Moderado" if prob_rain < 40 else "Alto")
                exec_text = (
                    f"**Meteo Running Pro - Dossier Técnico**\n"
                    f"**Data da Prova:** {race_day} de {race_month_name} \n**Local:** {city}\n\n"
                    f"Com base em {years_count} anos de dados (ERA5 / Open-Meteo), a data apresenta um Running Score Diário de **{avg_score:.0f}/100**. "
                    f"A probabilidade de chuva (>1mm) é de **{prob_rain:.1f}%** (risco {risk_level.lower()}). "
                    f"As temperaturas médias rondam os **{avg_temp:.1f}°C** (podendo atingir máximos históricos de {hottest['temp_max']:.1f}°C e mínimos de {coldest['temp_min']:.1f}°C). "
                    f"O vento máximo esperado é de **{avg_wind:.1f} km/h**."
                )
                if avg_sunrise:
                    exec_text += f" O nascer do sol acontece pelas {avg_sunrise} e o pôr do sol pelas {avg_sunset}."
            
                st.code(exec_text, language="markdown")
                st.download_button(
                    label="📥 Exportar Dossier em Texto (.txt)",
                    data=exec_text,
                    file_name=f"dossier_corrida_{city}_{race_day}_{race_month_name}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                # ── Advanced Hourly Simulations ────────────
            with tab_sim:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">🤖 Simulações Avançadas (Módulo Organização Pro)</div>', unsafe_allow_html=True)
                st.markdown(
                    '<p style="color:#b2bec3; font-size:0.95rem;">'
                    'Simuladores Horários de Performance, Logística (Reforços) e Risco Frontal (Aerodinâmica).</p>',
                    unsafe_allow_html=True,
                )

                with st.spinner("A carregar modelo horário (ERA5)..."):
                    years_to_fetch = list(range(year_range[0], year_range[1] + 1))
                    df_hr = fetch_hourly_specific_day(CITIES[city][0], CITIES[city][1], race_month_num, race_day, years_to_fetch)

                if not df_hr.empty:
                    df_hr["hr_score"] = df_hr.apply(compute_hourly_score, axis=1)

                    sim_c1, sim_c2, sim_c3 = st.columns(3)
                    with sim_c1:
                        hora_partida = st.slider("Hora Prevista da Partida:", min_value=6, max_value=20, value=9, step=1)
                    with sim_c2:
                        dist_opts = {"5 km": 5, "8 km": 8, "10 km": 10, "Meia-Maratona (21.1 km)": 21.1, "Maratona (42.2 km)": 42.2}
                        dist_nome = st.selectbox("Distância da Prova:", list(dist_opts.keys()), index=2)
                        dist_km = dist_opts[dist_nome]
                    with sim_c3:
                        dir_opts = {"Norte (0°)": 0, "Nordeste (45°)": 45, "Este (90°)": 90, "Sudeste (135°)": 135, "Sul (180°)": 180, "Sudoeste (225°)": 225, "Oeste (270°)": 270, "Noroeste (315°)": 315}
                        dir_nome = st.selectbox("O percurso corre para:", list(dir_opts.keys()))
                        dir_graus = dir_opts[dir_nome]
                    
                    # Dynamic duration based on distance to analyze weather during the event
                    if dist_km <= 5: duracao = 1
                    elif dist_km <= 10: duracao = 2
                    elif dist_km <= 22: duracao = 3
                    else: duracao = 5
                
                    # Filter conditions for the exact departure hour up to duration
                    df_prova = df_hr[(df_hr["hour"] >= hora_partida) & (df_hr["hour"] < hora_partida + duracao)]
                    if not df_prova.empty:
                        st.markdown(f"##### 🩺 Impacto Fisiológico Previsto ({dist_nome}):")
                    
                        media_app_temp = df_prova["apparent_temperature"].mean()
                        media_rad = df_prova["shortwave_radiation"].mean()
                        media_vento = df_prova["wind_speed_10m"].mean()
                        media_chuva = df_prova["precipitation"].mean()
                    
                        # 1. Performance Drop Simulator based on distance
                        multiplier = 0.5
                        if dist_km >= 42: multiplier = 2.0
                        elif dist_km >= 21: multiplier = 1.5
                        elif dist_km >= 10: multiplier = 1.0

                        if media_app_temp > 15:
                            quebra = (media_app_temp - 15) * multiplier
                            st.warning(f"📉 **Quebra de Performance ({dist_nome}):** A Sensação Térmica de {media_app_temp:.1f}°C causará uma quebra fisiológica média de **+{quebra:.1f}%** no tempo final dos atletas. (Modelos ACSM)")
                        elif media_app_temp < 5:
                            st.info(f"❄️ **Frio ({dist_nome}):** Sensação térmica de {media_app_temp:.1f}°C. O aquecimento pré-prova necessita ser prolongado. Risco de lesões musculares em ritmos rápidos.")
                        else:
                            st.success(f"🚀 **Performance Máxima ({dist_nome}):** Sensação térmica central de {media_app_temp:.1f}°C é absolutamente perfeita para bater recordes pessoais.")

                        # 2. Hydration/Logistics dynamically scaled by distance and heat
                        if media_app_temp > 22 or media_rad > 500:
                            if dist_km >= 21:
                                st.error(f"💧 **Logística Crítica ({dist_nome}):** Termorregulação sob stress severo permanente. Reforços a **cada 2.5km** obrigatórios. Risco muito elevado de insolação (Irradiação: {media_rad:.0f} W/m²).")
                            elif dist_km >= 10:
                                st.error(f"💧 **Logística Crítica ({dist_nome}):** Adicionar pontos extra de água. Risco de insolação e exaustão pelo calor apesar da distância.")
                            else:
                                st.warning(f"💦 **Logística de Calor ({dist_nome}):** Distância curta, mas devido ao calor extremo, exige 1 a 2 postos de água com capacidade de arrefecimento rápido.")
                        elif media_app_temp > 15:
                            if dist_km >= 21:
                                st.warning(f"💦 **Logística Essencial ({dist_nome}):** Reforços de 5 em 5km vão ser consumidos. Prever bebida isotónica na segunda metade da prova.")
                            else:
                                st.info(f"✅ **Logística Padrão ({dist_nome}):** Temperatura pede hidratação habitual, 1 a 2 postos são suficientes.")
                        else:
                            if dist_km >= 21:
                                st.info(f"✅ **Logística Aliviada ({dist_nome}):** Temperatura baixa inibe a desidratação severa. Água a cada 5km está perfeitamente marginada.")
                            else:
                                st.success(f"✅ **Hidratação Residual ({dist_nome}):** Clima frio/ótimo; postos base por protocolo são suficientes, pouca exigência na pista.")

                        # 3. Wind Direction / Headwind
                        def get_headwind_intensity(vento_media, graus_vento, graus_curso):
                            diff = abs(graus_vento - graus_curso)
                            if diff > 180: diff = 360 - diff
                            if diff < 45: return True, "Grave (Vento Frontal Direto)"
                            elif 45 <= diff <= 90: return True, "Moderado (Frontal-Lateral)"
                            return False, "Favorável ou Lateral Puro"

                        median_wind_dir = df_prova["wind_direction_10m"].median()
                        is_headwind, severity = get_headwind_intensity(media_vento, median_wind_dir, dir_graus)
                    
                        if is_headwind and media_vento > 10:
                            if dist_km >= 21:
                                st.error(f"🌬️ **Risco de Vento Frontal ({severity}):** Origem nos {median_wind_dir:.0f}°. Com prova longa para {dir_nome}, o choque aerodinâmico ({media_vento:.1f} km/h media) destruirá táticas de grupo.")
                            else:
                                st.warning(f"🌬️ **Vento Frontal ({severity}):** O vento de frente causará algum desconforto nos parciais ({media_vento:.1f} km/h).")
                        elif is_headwind:
                            st.info(f"🌬️ **Vento Frontal Ligeiro ({media_vento:.1f} km/h):** Impacto aerodinâmico baixo.")
                        else:
                            st.success(f"💨 **Bom Vento ({dist_nome}):** {severity}. Vento neutro ou pelas costas a ajudar o ritmo base da prova!")

                    # Hourly Departure Optimizer Chart
                    st.markdown("##### ⏱️ Otimizador: Qual a melhor hora de Partida?")
                    st.caption("Evolução estatística do 'Running Score' ao longo das horas do dia escolhido (em todos os anos).")
                    hourly_stats = df_hr.groupby("hour").agg(score_medio=("hr_score", "mean")).reset_index()
                
                    fig_hora = px.line(
                        hourly_stats, 
                        x="hour", 
                        y="score_medio", 
                        markers=True,
                        template="plotly_dark",
                        labels={"hour": "Hora do Dia (H)", "score_medio": "Running Score (0-100)"}
                    )
                    fig_hora.add_vline(x=hora_partida, line_dash="dash", line_color="springgreen", annotation_text="Sua Partida")
                    fig_hora.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1), yaxis_range=[0, 105], margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_hora, use_container_width=True, config={'displayModeBar': False})

                    # ── FEATURE 5: Janela Horária Ideal (Resumo Visual) ────────────
                    best_hour = hourly_stats.loc[hourly_stats["score_medio"].idxmax()]
                    worst_hour = hourly_stats.loc[hourly_stats["score_medio"].idxmin()]
                    # Find optimal window (consecutive hours with score > 80% of max)
                    threshold = best_hour["score_medio"] * 0.9
                    good_hours = hourly_stats[hourly_stats["score_medio"] >= threshold]["hour"].tolist()
                    if good_hours:
                        window_start = min(good_hours)
                        window_end = max(good_hours)
                        st.markdown(f'''
                        <div style="background: linear-gradient(135deg, rgba(0,200,83,0.15), rgba(0,200,83,0.03)); 
                                    border-left: 4px solid #00c853; border-radius: 8px; 
                                    padding: 15px 20px; margin: 15px 0;">
                            <div style="font-size: 1.1rem; font-weight: 700; color: #00c853;">⏱️ Janela Ideal de Partida</div>
                            <div style="font-size: 0.95rem; color: #dfe6e9; margin-top: 8px;">
                                A janela ideal de partida para esta data é entre as <strong>{int(window_start):02d}:00</strong> e as <strong>{int(window_end):02d}:00</strong> 
                                (Score médio: <strong>{best_hour["score_medio"]:.0f}/100</strong> às {int(best_hour["hour"]):02d}h).<br>
                                <span style="color: #ff6b6b;">Evitar partida às {int(worst_hour["hour"]):02d}:00</span> — Score cai para {worst_hour["score_medio"]:.0f}/100.
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)


            # ── FEATURE 6: Painel de Anomalias Históricas ────────────
            with tab_anomalies:
                st.markdown('<div class="section-header">🚨 Radar de Anomalias Históricas ("Cisnes Negros")</div>', unsafe_allow_html=True)
                st.markdown(
                    '<p style="color:#b2bec3; font-size:0.95rem;">'
                    'Eventos climatéricos atípicos e extremos registados nesta data ao longo do histórico. '
                    'Estes eventos não aparecem nas médias, mas representam riscos reais que o organizador deve considerar nos seus planos de contingência.</p>',
                    unsafe_allow_html=True,
                )

                # Calculate statistical thresholds using IQR method
                q75_rain = race_data["precipitation"].quantile(0.75)
                q25_rain = race_data["precipitation"].quantile(0.25)
                iqr_rain = q75_rain - q25_rain
                rain_threshold = q75_rain + 1.5 * iqr_rain if iqr_rain > 0 else race_data["precipitation"].mean() + race_data["precipitation"].std() * 2

                q75_wind = race_data["wind_max"].quantile(0.75)
                iqr_wind = q75_wind - race_data["wind_max"].quantile(0.25)
                wind_threshold = q75_wind + 1.5 * iqr_wind if iqr_wind > 0 else race_data["wind_max"].mean() + race_data["wind_max"].std() * 2

                q75_temp = race_data["temp_max"].quantile(0.75)
                iqr_temp = q75_temp - race_data["temp_max"].quantile(0.25)
                heat_threshold = q75_temp + 1.5 * iqr_temp if iqr_temp > 0 else race_data["temp_max"].mean() + race_data["temp_max"].std() * 2

                q25_cold = race_data["temp_min"].quantile(0.25)
                iqr_cold = race_data["temp_min"].quantile(0.75) - q25_cold
                cold_threshold = q25_cold - 1.5 * iqr_cold if iqr_cold > 0 else race_data["temp_min"].mean() - race_data["temp_min"].std() * 2

                anomalies_found = False

                # Rain anomalies
                rain_anomalies = race_data[race_data["precipitation"] > max(rain_threshold, 5)].sort_values("precipitation", ascending=False)
                if not rain_anomalies.empty:
                    anomalies_found = True
                    st.markdown("##### 🌧️ Eventos de Precipitação Extrema")
                    for _, row in rain_anomalies.iterrows():
                        severity = "SEVERO" if row["precipitation"] > 20 else "MODERADO"
                        sev_color = "#d32f2f" if severity == "SEVERO" else "#ff9800"
                        st.markdown(f'''
                        <div style="background: {sev_color}15; border-left: 3px solid {sev_color}; padding: 10px 15px; margin: 5px 0; border-radius: 4px;">
                            <strong style="color: {sev_color};">[{severity}]</strong> 
                            <strong>{row["date"].strftime("%d/%m/%Y")}</strong> — 
                            Precipitação de <strong>{row["precipitation"]:.1f} mm</strong> 
                            (a média histórica é {race_data["precipitation"].mean():.1f} mm). 
                            Score do dia: {row["running_score"]:.0f}/100
                        </div>
                        ''', unsafe_allow_html=True)

                # Wind anomalies
                wind_anomalies = race_data[race_data["wind_max"] > max(wind_threshold, 25)].sort_values("wind_max", ascending=False)
                if not wind_anomalies.empty:
                    anomalies_found = True
                    st.markdown("##### 🌬️ Eventos de Vento Extremo")
                    for _, row in wind_anomalies.iterrows():
                        severity = "SEVERO" if row["wind_max"] > 40 else "MODERADO"
                        sev_color = "#d32f2f" if severity == "SEVERO" else "#ff9800"
                        st.markdown(f'''
                        <div style="background: {sev_color}15; border-left: 3px solid {sev_color}; padding: 10px 15px; margin: 5px 0; border-radius: 4px;">
                            <strong style="color: {sev_color};">[{severity}]</strong> 
                            <strong>{row["date"].strftime("%d/%m/%Y")}</strong> — 
                            Rajada máxima de <strong>{row["wind_max"]:.1f} km/h</strong> 
                            (a média histórica é {race_data["wind_max"].mean():.1f} km/h). 
                            Impacto na estabilidade de estruturas e segurança.
                        </div>
                        ''', unsafe_allow_html=True)

                # Heat anomalies
                heat_anomalies = race_data[race_data["temp_max"] > max(heat_threshold, 30)].sort_values("temp_max", ascending=False)
                if not heat_anomalies.empty:
                    anomalies_found = True
                    st.markdown("##### 🔥 Eventos de Calor Extremo")
                    for _, row in heat_anomalies.iterrows():
                        severity = "SEVERO" if row["temp_max"] > 35 else "MODERADO"
                        sev_color = "#d32f2f" if severity == "SEVERO" else "#ff9800"
                        st.markdown(f'''
                        <div style="background: {sev_color}15; border-left: 3px solid {sev_color}; padding: 10px 15px; margin: 5px 0; border-radius: 4px;">
                            <strong style="color: {sev_color};">[{severity}]</strong> 
                            <strong>{row["date"].strftime("%d/%m/%Y")}</strong> — 
                            Temperatura máxima de <strong>{row["temp_max"]:.1f}°C</strong> 
                            (a média histórica é {race_data["temp_max"].mean():.1f}°C). 
                            Risco de insolação e exaustão.
                        </div>
                        ''', unsafe_allow_html=True)

                # Cold anomalies
                cold_anomalies = race_data[race_data["temp_min"] < min(cold_threshold, 2)].sort_values("temp_min", ascending=True)
                if not cold_anomalies.empty:
                    anomalies_found = True
                    st.markdown("##### 🧊 Eventos de Frio Extremo")
                    for _, row in cold_anomalies.iterrows():
                        severity = "SEVERO" if row["temp_min"] < 0 else "MODERADO"
                        sev_color = "#2196f3" if severity == "MODERADO" else "#9c27b0"
                        st.markdown(f'''
                        <div style="background: {sev_color}15; border-left: 3px solid {sev_color}; padding: 10px 15px; margin: 5px 0; border-radius: 4px;">
                            <strong style="color: {sev_color};">[{severity}]</strong> 
                            <strong>{row["date"].strftime("%d/%m/%Y")}</strong> — 
                            Temperatura mínima de <strong>{row["temp_min"]:.1f}°C</strong> 
                            (a média histórica é {race_data["temp_min"].mean():.1f}°C). 
                            Risco de hipotermia e lesão muscular.
                        </div>
                        ''', unsafe_allow_html=True)

                # Worst overall days
                bad_score_days = race_data[race_data["running_score"] < 40].sort_values("running_score", ascending=True)
                if not bad_score_days.empty:
                    anomalies_found = True
                    st.markdown("##### 💀 Dias com Score Crítico (<40/100)")
                    for _, row in bad_score_days.iterrows():
                        st.markdown(f'''
                        <div style="background: rgba(211,47,47,0.1); border-left: 3px solid #d32f2f; padding: 10px 15px; margin: 5px 0; border-radius: 4px;">
                            <strong style="color: #d32f2f;">[CRÍTICO]</strong> 
                            <strong>{row["date"].strftime("%d/%m/%Y")}</strong> — 
                            Score: <strong>{row["running_score"]:.0f}/100</strong>. 
                            Temp: {row["temp_max"]:.1f}°C/{row["temp_min"]:.1f}°C, 
                            Chuva: {row["precipitation"]:.1f}mm, 
                            Vento: {row["wind_max"]:.1f}km/h
                        </div>
                        ''', unsafe_allow_html=True)

                if not anomalies_found:
                    st.success("✅ **Sem anomalias relevantes registadas!** O histórico desta data não apresenta eventos climatéricos extremos. Risco de 'cisne negro' muito baixo.")
                else:
                    total_anomalies = len(rain_anomalies) + len(wind_anomalies) + len(heat_anomalies) + len(cold_anomalies) + len(bad_score_days)
                    pct_anomaly = (total_anomalies / years_count) * 100
                    st.markdown(f'''
                    <div style="background: rgba(255,152,0,0.1); border: 1px solid rgba(255,152,0,0.3); border-radius: 8px; padding: 12px 18px; margin-top: 15px;">
                        <strong>📊 Resumo:</strong> Foram detetados <strong>{total_anomalies} eventos atípicos</strong> em {years_count} anos de histórico 
                        ({pct_anomaly:.0f}% dos anos). Recomenda-se que o plano de contingência da prova cubra especificamente estes cenários.
                    </div>
                    ''', unsafe_allow_html=True)

            with tab_hist:
                st.markdown("#### 📅 Tabela de Dados Históricos Ocorridos")
                display_cols = ["date", "temp_max", "temp_min", "precipitation", "wind_max", "running_score"]
                rename_map = {
                    "date": "Data", "temp_max": "Temp Máx (°C)", "temp_min": "Temp Mín (°C)", 
                    "precipitation": "Precipitação (mm)", "wind_max": "Vento Máx (km/h)", "running_score": "Score"
                }
                if race_data["sunrise"].notna().any():
                    race_display = race_data[display_cols + ["sunrise", "sunset"]].copy()
                    race_display["sunrise"] = race_display["sunrise"].dt.strftime("%H:%M")
                    race_display["sunset"]  = race_display["sunset"].dt.strftime("%H:%M")
                    rename_map["sunrise"] = "Nascer Sol"
                    rename_map["sunset"]  = "Pôr Sol"
                else:
                    race_display = race_data[display_cols].copy()
                st.dataframe(
                    race_display.sort_values("date", ascending=False).rename(columns=rename_map),
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.warning("Sem dados suficientes para este intervalo.")

    with tab_compare:
        # ── Date/City Comparator ───────────────────────
        st.markdown('<div class="section-header">⚖️ Comparador Pro de Cenários</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#b2bec3; font-size:0.95rem;">'
            'Compare até 3 datas ou 3 cidades lado a lado. Defina as especificidades da sua prova e obtenha '
            'um relatório detalhado com conclusões, sugestão de hora de partida, impacto fisiológico e recomendações operacionais.</p>',
            unsafe_allow_html=True,
        )

        # ── Race Specifications ───────────────────────
        st.markdown("##### 🏁 Especificidades da Prova")
        spec_c1, spec_c2, spec_c3 = st.columns(3)
        with spec_c1:
            cmp_dist_opts = {"5 km": 5, "8 km": 8, "10 km": 10, "Meia-Maratona (21.1 km)": 21.1, "Maratona (42.2 km)": 42.2}
            cmp_dist_nome = st.selectbox("Distância da Prova:", list(cmp_dist_opts.keys()), index=2, key="cmp_dist")
            cmp_dist_km = cmp_dist_opts[cmp_dist_nome]
        with spec_c2:
            cmp_hora_partida = st.slider("Hora Prevista da Partida:", min_value=6, max_value=20, value=9, step=1, key="cmp_hora")
        with spec_c3:
            cmp_dir_opts = {"Norte (0°)": 0, "Nordeste (45°)": 45, "Este (90°)": 90, "Sudeste (135°)": 135, "Sul (180°)": 180, "Sudoeste (225°)": 225, "Oeste (270°)": 270, "Noroeste (315°)": 315}
            cmp_dir_nome = st.selectbox("Direcção do Percurso:", list(cmp_dir_opts.keys()), key="cmp_dir")
            cmp_dir_graus = cmp_dir_opts[cmp_dir_nome]

        # Duration based on distance
        if cmp_dist_km <= 5: cmp_duracao = 1
        elif cmp_dist_km <= 10: cmp_duracao = 2
        elif cmp_dist_km <= 22: cmp_duracao = 3
        else: cmp_duracao = 5

        st.divider()

        compare_mode = st.radio("Modo de Comparação:", ["Múltiplas Datas (Mesma Cidade)", "Múltiplas Cidades (Mesma Data)"], horizontal=True, key="comp_mode")

        # Helper function to generate report for a scenario
        def generate_scenario_report(cdata_daily, df_hourly, label, city_name, dist_km, dist_nome, hora_partida, duracao, dir_graus, dir_nome):
            """Generate a detailed report dict for a comparison scenario."""
            report = {"label": label, "city": city_name}

            if cdata_daily.empty:
                report["valid"] = False
                return report
            report["valid"] = True

            # Basic daily stats (fallback)
            years_count = cdata_daily["year"].nunique()
            avg_score_daily = cdata_daily["running_score"].mean()
            prob_rain = (cdata_daily["precipitation"] > 1).mean() * 100
            avg_temp = cdata_daily["temp_avg"].mean()
            avg_app_temp = cdata_daily["app_temp_avg"].mean()
            avg_wind = cdata_daily["wind_max"].mean()
            max_temp_hist = cdata_daily["temp_max"].max()
            min_temp_hist = cdata_daily["temp_min"].min()
            
            # If we have hourly data, evaluate stats specifically during the planned race hours!
            if df_hourly is not None and not df_hourly.empty:
                df_race_hours = df_hourly[(df_hourly["hour"] >= hora_partida) & (df_hourly["hour"] < hora_partida + duracao)].copy()
                if not df_race_hours.empty:
                    df_race_hours["hr_score"] = df_race_hours.apply(compute_hourly_score, axis=1)
                    avg_score = df_race_hours["hr_score"].mean()
                    # Probability of rain during these specific hours
                    prob_rain = (df_race_hours["precipitation"] > 0).groupby(df_race_hours["time"].dt.date).any().mean() * 100
                    avg_temp = df_race_hours["temperature_2m"].mean()
                    avg_app_temp = df_race_hours["apparent_temperature"].mean()
                    avg_wind = df_race_hours["wind_speed_10m"].mean()
                else:
                    avg_score = avg_score_daily
            else:
                avg_score = avg_score_daily

            report["years"] = years_count
            report["score"] = avg_score
            report["prob_rain"] = prob_rain
            report["avg_temp"] = avg_temp
            report["avg_app_temp"] = avg_app_temp
            report["avg_wind"] = avg_wind
            report["max_temp_hist"] = max_temp_hist
            report["min_temp_hist"] = min_temp_hist

            # Sunrise/sunset
            report["sunrise"] = "-"
            report["sunset"] = "-"
            report["daylight"] = "-"
            if cdata_daily["sunrise"].notna().any():
                sr_s = cdata_daily["sunrise"].dropna().apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second).mean()
                report["sunrise"] = f"{int(sr_s // 3600):02d}:{int((sr_s % 3600) // 60):02d}"
                ss_s = cdata_daily["sunset"].dropna().apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second).mean()
                report["sunset"] = f"{int(ss_s // 3600):02d}:{int((ss_s % 3600) // 60):02d}"
                report["daylight"] = f"{cdata_daily['daylight_hours'].mean():.1f}h"

            # Risk semaphore
            risk_score = 0
            risk_factors = []
            if avg_temp > 28 or avg_temp < 2: risk_score += 3; risk_factors.append("Temperatura extrema")
            elif avg_temp > 22 or avg_temp < 5: risk_score += 2; risk_factors.append("Temperatura desfavorável")
            elif avg_temp > 18 or avg_temp < 8: risk_score += 1
            if prob_rain > 50: risk_score += 3; risk_factors.append("Chuva muito provável")
            elif prob_rain > 30: risk_score += 2; risk_factors.append("Chuva moderada")
            elif prob_rain > 15: risk_score += 1
            if avg_wind > 30: risk_score += 3; risk_factors.append("Vento forte")
            elif avg_wind > 20: risk_score += 2; risk_factors.append("Vento moderado")
            elif avg_wind > 12: risk_score += 1
            report["risk_score"] = risk_score
            report["risk_factors"] = risk_factors

            # Performance impact
            multiplier = 0.5
            if dist_km >= 42: multiplier = 2.0
            elif dist_km >= 21: multiplier = 1.5
            elif dist_km >= 10: multiplier = 1.0
            if avg_app_temp > 15:
                report["perf_drop"] = (avg_app_temp - 15) * multiplier
            else:
                report["perf_drop"] = 0

            # Hourly analysis - optimal window
            report["best_window"] = None
            report["best_hour_score"] = None
            report["worst_hour"] = None
            report["worst_hour_score"] = None
            if df_hourly is not None and not df_hourly.empty:
                df_hourly["hr_score"] = df_hourly.apply(compute_hourly_score, axis=1)
                hourly_stats = df_hourly.groupby("hour").agg(score_medio=("hr_score", "mean")).reset_index()
                # Only consider reasonable running hours (6h-20h)
                hourly_running = hourly_stats[(hourly_stats["hour"] >= 6) & (hourly_stats["hour"] <= 20)].copy()
                if not hourly_running.empty:
                    best_h = hourly_running.loc[hourly_running["score_medio"].idxmax()]
                    worst_h = hourly_running.loc[hourly_running["score_medio"].idxmin()]
                    # Tighter threshold: top 95% of best score
                    threshold = best_h["score_medio"] * 0.95
                    good_hours = sorted(hourly_running[hourly_running["score_medio"] >= threshold]["hour"].astype(int).tolist())
                    # Find best contiguous block
                    if good_hours:
                        blocks = []
                        current_block = [good_hours[0]]
                        for h in good_hours[1:]:
                            if h == current_block[-1] + 1:
                                current_block.append(h)
                            else:
                                blocks.append(current_block)
                                current_block = [h]
                        blocks.append(current_block)
                        # Pick the longest contiguous block (or first if tied)
                        best_block = max(blocks, key=len)
                        report["best_window"] = f"{best_block[0]:02d}:00 - {best_block[-1]+1:02d}:00"
                        report["best_hour_score"] = best_h["score_medio"]
                        report["worst_hour"] = int(worst_h["hour"])
                        report["worst_hour_score"] = worst_h["score_medio"]

                    # Wind analysis at race time
                    df_race_hours = df_hourly[(df_hourly["hour"] >= hora_partida) & (df_hourly["hour"] < hora_partida + duracao)]
                    if not df_race_hours.empty:
                        race_wind = df_race_hours["wind_speed_10m"].mean()
                        race_wind_dir = df_race_hours["wind_direction_10m"].median()
                        diff = abs(race_wind_dir - dir_graus)
                        if diff > 180: diff = 360 - diff
                        report["headwind"] = diff < 90 and race_wind > 10
                        report["race_wind"] = race_wind
                        report["race_wind_dir"] = race_wind_dir

            # Conclusions
            conclusions = []
            conclusions.append(f"📌 Análise focada na hora prevista: {hora_partida:02d}:00")
            
            if risk_score <= 2:
                conclusions.append("✅ Data segura para a realização da prova")
            elif risk_score <= 5:
                conclusions.append("⚠️ Data viável mas requer plano de contingência reforçado")
            else:
                conclusions.append("🔴 Data de alto risco — considerar alternativa")

            if report["perf_drop"] > 5:
                conclusions.append(f"📉 Quebra de performance estimada: +{report['perf_drop']:.1f}% no tempo final")
            elif report["perf_drop"] > 0:
                conclusions.append(f"📊 Ligeira quebra de performance: +{report['perf_drop']:.1f}%")
            else:
                conclusions.append("🚀 Condições ideais para performance máxima")

            if prob_rain > 30:
                conclusions.append("🌧 Prever zonas de refúgio e sinalização de piso molhado")
            if avg_wind > 20:
                conclusions.append("💨 Reavaliar fixação de estruturas e infláveis")
            if report.get("headwind") and report.get("race_wind", 0) > 15:
                conclusions.append(f"🌬️ Vento frontal previsto ({report['race_wind']:.0f} km/h) — impacto nos tempos")

            report["conclusions"] = conclusions
            return report

        def create_pdf_report(reports, title):
            def sanitize(text):
                # encode to latin-1 ignoring errors (drops emojis entirely) so FPDF text wrapper doesn't break
                return text.encode('latin-1', 'ignore').decode('latin-1')
                
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", style="B", size=16)
            pdf.cell(0, 10, sanitize(title), ln=True, align="C")
            pdf.ln(5)
            
            for i, rpt in enumerate(reports):
                if not rpt.get("valid"):
                    continue
                pdf.set_font("Helvetica", style="B", size=14)
                pdf.cell(0, 10, sanitize(f"Cenario {i+1}: {rpt['label']}"), ln=True)
                pdf.set_font("Helvetica", size=11)
                
                pdf.cell(0, 8, sanitize(f"Score Geral: {rpt['score']:.0f}/100"), ln=True)
                
                risk_s = rpt["risk_score"]
                sem_label = "SEGURA" if risk_s <= 2 else ("PRECAUCAO" if risk_s <= 5 else "CRITICA")
                pdf.cell(0, 8, sanitize(f"Risco: {sem_label}"), ln=True)
                
                pdf.cell(0, 8, sanitize(f"Probabilidade de Chuva: {rpt['prob_rain']:.0f}%"), ln=True)
                pdf.cell(0, 8, sanitize(f"Temperatura Media: {rpt['avg_temp']:.1f}C (Sensacao: {rpt['avg_app_temp']:.1f}C)"), ln=True)
                pdf.cell(0, 8, sanitize(f"Vento Medio: {rpt['avg_wind']:.1f} km/h"), ln=True)
                
                window_txt = rpt["best_window"] if rpt["best_window"] else "-"
                pdf.cell(0, 8, sanitize(f"Janela Ideal de Partida: {window_txt}"), ln=True)
                pdf.cell(0, 8, sanitize(f"Quebra de Performance Estimada: +{rpt['perf_drop']:.1f}%"), ln=True)
                
                pdf.ln(2)
                pdf.set_font("Helvetica", style="B", size=11)
                pdf.cell(0, 8, "Conclusoes:", ln=True)
                pdf.set_font("Helvetica", size=11)
                for conc in rpt["conclusions"]:
                    pdf.multi_cell(0, 6, sanitize(f"- {conc}"), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(5)
                
            return pdf.output()

        if "Datas" in compare_mode:
            num_dates = st.radio("Quantas datas comparar?", [2, 3], horizontal=True, key="num_compare_dates")
            compare_cols = st.columns(num_dates)
            compare_dates = []
            for i, col in enumerate(compare_cols):
                with col:
                    st.markdown(f"**Data {i+1}**")
                    cm = st.selectbox(f"Mês", list(MONTH_NAMES_PT.values()), key=f"cmp_month_{i}", index=i * 3)
                    cm_num = name_to_num[cm]
                    sd_m = pd.to_datetime(f"2024-{cm_num:02}-01")
                    cd = st.number_input(f"Dia", min_value=1, max_value=sd_m.days_in_month, value=min(15, sd_m.days_in_month), step=1, key=f"cmp_day_{i}")
                    compare_dates.append((cm_num, cd, f"{cd} {cm}"))

            if st.button("🔄 Comparar Datas", type="primary", use_container_width=True):
                reports = []
                with st.spinner("A analisar cenários e gerar relatório completo..."):
                    for month_n, day_n, label in compare_dates:
                        cdata = df[
                            (df["month"] == month_n) & 
                            (df["day"] == day_n) & 
                            (df["year"] >= year_range[0]) & 
                            (df["year"] <= year_range[1])
                        ]
                        # Fetch hourly data
                        years_list = list(range(year_range[0], year_range[1] + 1))
                        df_hr = fetch_hourly_specific_day(CITIES[city][0], CITIES[city][1], month_n, day_n, years_list)
                        report = generate_scenario_report(cdata, df_hr, f"{label} ({city})", city, cmp_dist_km, cmp_dist_nome, cmp_hora_partida, cmp_duracao, cmp_dir_graus, cmp_dir_nome)
                        reports.append(report)

                # Find winner
                valid_scores = [(i, r["score"]) for i, r in enumerate(reports) if r.get("valid")]
                best_idx = max(valid_scores, key=lambda x: x[1])[0] if valid_scores else 0

                # Display visual summary cards
                st.markdown("##### 📊 Resumo Comparativo")
                card_cols = st.columns(len(reports))
                for idx, (col, rpt) in enumerate(zip(card_cols, reports)):
                    with col:
                        if not rpt.get("valid"):
                            st.warning("Sem dados")
                            continue
                        is_best = idx == best_idx
                        border_color = "#00c853" if is_best else "rgba(255,255,255,0.1)"
                        bg = "rgba(0,200,83,0.08)" if is_best else "rgba(255,255,255,0.02)"
                        score_color = "#00c853" if is_best else "#74b9ff"
                        badge = '<div style="background:#00c853;color:#fff;padding:3px 12px;border-radius:12px;font-size:0.75rem;font-weight:700;display:inline-block;margin-bottom:8px;">🏆 MELHOR OPÇÃO</div>' if is_best else ""
                        c_label = rpt["label"]
                        c_score = f"{rpt['score']:.0f}"
                        c_risk = rpt["risk_score"]
                        c_sem_icon = "🟢" if c_risk <= 2 else ("🟡" if c_risk <= 5 else "🔴")
                        c_sem_label = "SEGURA" if c_risk <= 2 else ("PRECAUÇÃO" if c_risk <= 5 else "CRÍTICA")
                        c_window = rpt["best_window"] if rpt["best_window"] else "-"
                        c_rain = f"{rpt['prob_rain']:.0f}"
                        c_temp = f"{rpt['avg_temp']:.1f}"
                        c_feel = f"{rpt['avg_app_temp']:.1f}"
                        c_wind = f"{rpt['avg_wind']:.1f}"
                        c_perf = f"{rpt['perf_drop']:.1f}"
                        c_sunrise = rpt["sunrise"]
                        c_sunset = rpt["sunset"]
                        card_html = (
                            f'<div style="background:{bg};border:2px solid {border_color};border-radius:12px;padding:18px;text-align:center;">'
                            f'{badge}'
                            f'<div style="font-size:1.05rem;font-weight:700;color:#dfe6e9;margin-bottom:8px;">{c_label}</div>'
                            f'<div style="font-size:2.2rem;font-weight:800;color:{score_color};margin:4px 0;">{c_score}/100</div>'
                            f'<div style="font-size:0.8rem;margin:6px 0;">{c_sem_icon} {c_sem_label}</div>'
                            f'<div style="font-size:0.82rem;color:rgba(255,255,255,0.55);line-height:1.9;margin-top:8px;">'
                            f'🌧 Chuva: {c_rain}%<br>'
                            f'🌡 Temp: {c_temp}°C<br>'
                            f'🤒 Sensação: {c_feel}°C<br>'
                            f'💨 Vento: {c_wind} km/h<br>'
                            f'⏱ Janela Ideal: {c_window}<br>'
                            f'📉 Quebra Perf.: +{c_perf}%<br>'
                            f'🌅 Nascer: {c_sunrise} · Pôr: {c_sunset}'
                            f'</div></div>'
                        )
                        st.markdown(card_html, unsafe_allow_html=True)

                # Detailed report per scenario
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 📋 Relatório Detalhado por Cenário")
                for idx, rpt in enumerate(reports):
                    if not rpt.get("valid"):
                        continue
                    is_best = idx == best_idx
                    header_color = "#00c853" if is_best else "#74b9ff"
                    trophy = " 🏆" if is_best else ""
                    with st.expander(f"{'🏆 ' if is_best else ''}📄 {rpt['label']} — Score: {rpt['score']:.0f}/100{trophy}", expanded=is_best):
                        # Semaphore
                        risk_s = rpt["risk_score"]
                        if risk_s <= 2:
                            sem_c, sem_l = "#00c853", "SEGURA"
                        elif risk_s <= 5:
                            sem_c, sem_l = "#ff9800", "PRECAUÇÃO"
                        else:
                            sem_c, sem_l = "#d32f2f", "CRÍTICA"
                        st.markdown(f'<div style="background:{sem_c}15; border-left:4px solid {sem_c}; padding:10px 15px; border-radius:4px; margin-bottom:12px;"><strong style="color:{sem_c};">🚦 Data {sem_l}</strong> — Índice de Risco: {risk_s}/9</div>', unsafe_allow_html=True)

                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Score", f"{rpt['score']:.0f}/100")
                        mc2.metric("Chuva >1mm", f"{rpt['prob_rain']:.0f}%")
                        mc3.metric("Temp. Média", f"{rpt['avg_temp']:.1f}°C")
                        mc4.metric("Vento Médio", f"{rpt['avg_wind']:.1f} km/h")

                        # Optimal window
                        if rpt["best_window"]:
                            st.markdown(f'''
                            <div style="background:rgba(0,200,83,0.08); border-left:4px solid #00c853; padding:10px 15px; border-radius:4px; margin:8px 0;">
                                <strong style="color:#00c853;">⏱️ Janela Ideal de Partida: {rpt["best_window"]}</strong><br>
                                <span style="color:#b2bec3; font-size:0.85rem;">
                                    Melhor hora: Score {rpt["best_hour_score"]:.0f}/100 · 
                                    Evitar {rpt["worst_hour"]:02d}:00 (Score: {rpt["worst_hour_score"]:.0f}/100)
                                </span>
                            </div>
                            ''', unsafe_allow_html=True)

                        # Performance impact
                        if rpt["perf_drop"] > 0:
                            perf_color = "#d32f2f" if rpt["perf_drop"] > 5 else "#ff9800"
                            st.markdown(f'<div style="background:{perf_color}15; border-left:4px solid {perf_color}; padding:10px 15px; border-radius:4px; margin:8px 0;"><strong style="color:{perf_color};">📉 Quebra de Performance ({cmp_dist_nome}): +{rpt["perf_drop"]:.1f}%</strong><br><span style="color:#b2bec3; font-size:0.85rem;">Sensação Térmica: {rpt["avg_app_temp"]:.1f}°C · Modelo ACSM</span></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div style="background:rgba(0,200,83,0.08); border-left:4px solid #00c853; padding:10px 15px; border-radius:4px; margin:8px 0;"><strong style="color:#00c853;">🚀 Performance Máxima ({cmp_dist_nome})</strong><br><span style="color:#b2bec3; font-size:0.85rem;">Sensação Térmica ideal: {rpt["avg_app_temp"]:.1f}°C</span></div>', unsafe_allow_html=True)

                        # Extreme records
                        st.markdown(f"<div style='color:#b2bec3; font-size:0.85rem; margin-top:8px;'>📊 Extremos históricos ({rpt['years']} anos): Máx {rpt['max_temp_hist']:.1f}°C · Mín {rpt['min_temp_hist']:.1f}°C</div>", unsafe_allow_html=True)

                        # Conclusions
                        st.markdown("**🎯 Conclusões:**")
                        for conclusion in rpt["conclusions"]:
                            st.markdown(f"- {conclusion}")

                # Final recommendation
                st.markdown("<br>", unsafe_allow_html=True)
                winner = reports[best_idx]
                rec_text = f"🏆 **Recomendação Final ({cmp_dist_nome}):** **{winner['label']}** com Score de **{winner['score']:.0f}/100**"
                if winner.get("best_window"):
                    rec_text += f" · Partida ideal: **{winner['best_window']}**"
                if winner["perf_drop"] > 0:
                    rec_text += f" · Quebra estimada: +{winner['perf_drop']:.1f}%"
                st.success(rec_text)
                
                if any(r.get("valid") for r in reports):
                    pdf_bytes = create_pdf_report(reports, "Comparacao de Datas - Meteo Running Pro")
                    st.download_button(
                        label="📄 Exportar Relatório PDF",
                        data=pdf_bytes,
                        file_name="comparador_datas.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

        else:
            # Multi-city mode
            num_cities = st.radio("Quantas cidades comparar?", [2, 3], horizontal=True, key="num_compare_cities")
            
            cdate_col1, cdate_col2 = st.columns(2)
            with cdate_col1:
                cm = st.selectbox(f"Mês da Prova", list(MONTH_NAMES_PT.values()), key="cmp_city_month", index=9)
                cm_num = name_to_num[cm]
            with cdate_col2:
                sd_m = pd.to_datetime(f"2024-{cm_num:02}-01")
                cd = st.number_input(f"Dia da Prova", min_value=1, max_value=sd_m.days_in_month, value=min(15, sd_m.days_in_month), step=1, key="cmp_city_day")
            
            compare_city_cols = st.columns(num_cities)
            compare_cities = []
            for i, col in enumerate(compare_city_cols):
                with col:
                    selected_city = st.selectbox(f"Cidade {i+1}", sorted(list(CITIES.keys())), key=f"cmp_city_{i}", index=i)
                    compare_cities.append(selected_city)
            
            if st.button("🔄 Comparar Cidades", type="primary", use_container_width=True):
                reports = []
                with st.spinner("A analisar cenários e gerar relatório completo..."):
                    for c in compare_cities:
                        lat_c, lon_c = CITIES[c]
                        c_df = fetch_weather_data(lat_c, lon_c, f"{year_range[0]}-01-01", f"{year_range[1]}-12-31")
                        c_df = add_scores(c_df)
                        cdata = c_df[(c_df["month"] == cm_num) & (c_df["day"] == cd)]
                        years_list = list(range(year_range[0], year_range[1] + 1))
                        df_hr = fetch_hourly_specific_day(lat_c, lon_c, cm_num, cd, years_list)
                        report = generate_scenario_report(cdata, df_hr, f"{c} ({cd}/{cm_num:02})", c, cmp_dist_km, cmp_dist_nome, cmp_hora_partida, cmp_duracao, cmp_dir_graus, cmp_dir_nome)
                        reports.append(report)

                # Find winner
                valid_scores = [(i, r["score"]) for i, r in enumerate(reports) if r.get("valid")]
                best_idx = max(valid_scores, key=lambda x: x[1])[0] if valid_scores else 0

                # Display visual summary cards
                st.markdown("##### 📊 Resumo Comparativo")
                card_cols = st.columns(len(reports))
                for idx, (col, rpt) in enumerate(zip(card_cols, reports)):
                    with col:
                        if not rpt.get("valid"):
                            st.warning("Sem dados")
                            continue
                        is_best = idx == best_idx
                        border_color = "#00c853" if is_best else "rgba(255,255,255,0.1)"
                        bg = "rgba(0,200,83,0.08)" if is_best else "rgba(255,255,255,0.02)"
                        score_color = "#00c853" if is_best else "#74b9ff"
                        badge = '<div style="background:#00c853;color:#fff;padding:3px 12px;border-radius:12px;font-size:0.75rem;font-weight:700;display:inline-block;margin-bottom:8px;">🏆 MELHOR LOCAL</div>' if is_best else ""
                        c_label = rpt["label"]
                        c_score = f"{rpt['score']:.0f}"
                        c_risk = rpt["risk_score"]
                        c_sem_icon = "🟢" if c_risk <= 2 else ("🟡" if c_risk <= 5 else "🔴")
                        c_sem_label = "SEGURA" if c_risk <= 2 else ("PRECAUÇÃO" if c_risk <= 5 else "CRÍTICA")
                        c_window = rpt["best_window"] if rpt["best_window"] else "-"
                        c_rain = f"{rpt['prob_rain']:.0f}"
                        c_temp = f"{rpt['avg_temp']:.1f}"
                        c_feel = f"{rpt['avg_app_temp']:.1f}"
                        c_wind = f"{rpt['avg_wind']:.1f}"
                        c_perf = f"{rpt['perf_drop']:.1f}"
                        c_sunrise = rpt["sunrise"]
                        c_sunset = rpt["sunset"]
                        card_html = (
                            f'<div style="background:{bg};border:2px solid {border_color};border-radius:12px;padding:18px;text-align:center;">'
                            f'{badge}'
                            f'<div style="font-size:1.05rem;font-weight:700;color:#dfe6e9;margin-bottom:8px;">{c_label}</div>'
                            f'<div style="font-size:2.2rem;font-weight:800;color:{score_color};margin:4px 0;">{c_score}/100</div>'
                            f'<div style="font-size:0.8rem;margin:6px 0;">{c_sem_icon} {c_sem_label}</div>'
                            f'<div style="font-size:0.82rem;color:rgba(255,255,255,0.55);line-height:1.9;margin-top:8px;">'
                            f'🌧 Chuva: {c_rain}%<br>'
                            f'🌡 Temp: {c_temp}°C<br>'
                            f'🤒 Sensação: {c_feel}°C<br>'
                            f'💨 Vento: {c_wind} km/h<br>'
                            f'⏱ Janela Ideal: {c_window}<br>'
                            f'📉 Quebra Perf.: +{c_perf}%<br>'
                            f'🌅 Nascer: {c_sunrise} · Pôr: {c_sunset}'
                            f'</div></div>'
                        )
                        st.markdown(card_html, unsafe_allow_html=True)

                # Detailed report per city
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 📋 Relatório Detalhado por Cidade")
                for idx, rpt in enumerate(reports):
                    if not rpt.get("valid"):
                        continue
                    is_best = idx == best_idx
                    trophy = " 🏆" if is_best else ""
                    with st.expander(f"{'🏆 ' if is_best else ''}📄 {rpt['label']} — Score: {rpt['score']:.0f}/100{trophy}", expanded=is_best):
                        risk_s = rpt["risk_score"]
                        if risk_s <= 2:
                            sem_c, sem_l = "#00c853", "SEGURA"
                        elif risk_s <= 5:
                            sem_c, sem_l = "#ff9800", "PRECAUÇÃO"
                        else:
                            sem_c, sem_l = "#d32f2f", "CRÍTICA"
                        st.markdown(f'<div style="background:{sem_c}15; border-left:4px solid {sem_c}; padding:10px 15px; border-radius:4px; margin-bottom:12px;"><strong style="color:{sem_c};">🚦 Data {sem_l}</strong> — Índice de Risco: {risk_s}/9</div>', unsafe_allow_html=True)

                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Score", f"{rpt['score']:.0f}/100")
                        mc2.metric("Chuva >1mm", f"{rpt['prob_rain']:.0f}%")
                        mc3.metric("Temp. Média", f"{rpt['avg_temp']:.1f}°C")
                        mc4.metric("Vento Médio", f"{rpt['avg_wind']:.1f} km/h")

                        if rpt["best_window"]:
                            st.markdown(f'''
                            <div style="background:rgba(0,200,83,0.08); border-left:4px solid #00c853; padding:10px 15px; border-radius:4px; margin:8px 0;">
                                <strong style="color:#00c853;">⏱️ Janela Ideal de Partida: {rpt["best_window"]}</strong><br>
                                <span style="color:#b2bec3; font-size:0.85rem;">
                                    Melhor hora: Score {rpt["best_hour_score"]:.0f}/100 · 
                                    Evitar {rpt["worst_hour"]:02d}:00 (Score: {rpt["worst_hour_score"]:.0f}/100)
                                </span>
                            </div>
                            ''', unsafe_allow_html=True)

                        if rpt["perf_drop"] > 0:
                            perf_color = "#d32f2f" if rpt["perf_drop"] > 5 else "#ff9800"
                            st.markdown(f'<div style="background:{perf_color}15; border-left:4px solid {perf_color}; padding:10px 15px; border-radius:4px; margin:8px 0;"><strong style="color:{perf_color};">📉 Quebra de Performance ({cmp_dist_nome}): +{rpt["perf_drop"]:.1f}%</strong><br><span style="color:#b2bec3; font-size:0.85rem;">Sensação Térmica: {rpt["avg_app_temp"]:.1f}°C · Modelo ACSM</span></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div style="background:rgba(0,200,83,0.08); border-left:4px solid #00c853; padding:10px 15px; border-radius:4px; margin:8px 0;"><strong style="color:#00c853;">🚀 Performance Máxima ({cmp_dist_nome})</strong><br><span style="color:#b2bec3; font-size:0.85rem;">Sensação Térmica ideal: {rpt["avg_app_temp"]:.1f}°C</span></div>', unsafe_allow_html=True)

                        st.markdown(f"<div style='color:#b2bec3; font-size:0.85rem; margin-top:8px;'>📊 Extremos históricos ({rpt['years']} anos): Máx {rpt['max_temp_hist']:.1f}°C · Mín {rpt['min_temp_hist']:.1f}°C</div>", unsafe_allow_html=True)

                        st.markdown("**🎯 Conclusões:**")
                        for conclusion in rpt["conclusions"]:
                            st.markdown(f"- {conclusion}")

                # Final recommendation
                st.markdown("<br>", unsafe_allow_html=True)
                winner = reports[best_idx]
                rec_text = f"🏆 **Recomendação Final ({cmp_dist_nome}):** **{winner['label']}** com Score de **{winner['score']:.0f}/100**"
                if winner.get("best_window"):
                    rec_text += f" · Partida ideal: **{winner['best_window']}**"
                if winner["perf_drop"] > 0:
                    rec_text += f" · Quebra estimada: +{winner['perf_drop']:.1f}%"
                st.success(rec_text)

                if any(r.get("valid") for r in reports):
                    pdf_bytes = create_pdf_report(reports, "Comparacao de Cidades - Meteo Running Pro")
                    st.download_button(
                        label="📄 Exportar Relatório PDF",
                        data=pdf_bytes,
                        file_name="comparador_cidades.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

    with tab_data:
        # ── CSV Export ────────────────────────────
        st.markdown('<div class="section-header">💾 Exportar Dados Filtrados</div>', unsafe_allow_html=True)

        export_df = df_filtered[["date", "temp_max", "temp_min", "temp_avg", "precipitation", "wind_max", "running_score"]].copy()
        export_df.columns = ["Data", "Temp. Máx (°C)", "Temp. Mín (°C)", "Temp. Média (°C)", "Precipitação (mm)", "Vento Máx (km/h)", "Running Score"]

        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Exportar CSV",
            data=csv,
            file_name=f"meteo_running_{city.lower()}_{year_range[0]}-{year_range[1]}.csv",
            mime="text/csv",
        )

    # ── Footer ───────────────────────────────
    st.markdown(
        '<div class="footer">'
        'Dados: ERA5 Reanalysis (ECMWF) via Open-Meteo API · Modelos Fisiológicos: ACSM Guidelines<br>'
        'Meteo Running Pro · Weather Decision Support System · Portugal'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
