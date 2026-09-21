from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="World Bank Life Expectancy & ML Forecasting",
    page_icon="📈",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed_data"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
EVAL_PATH = RESULTS_DIR / "forecast_catboost.csv"


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    life = pd.read_csv(DATA_DIR / "processed_life_expectancy_total.csv")
    male = pd.read_csv(DATA_DIR / "processed_life_expectancy_male.csv")
    female = pd.read_csv(DATA_DIR / "processed_life_expectancy_female.csv")
    fertility = pd.read_csv(DATA_DIR / "processed_fertility_rate.csv")
    death = pd.read_csv(DATA_DIR / "processed_death_rate.csv")

    for df in [life, male, female, fertility, death]:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    life = life[["country_name", "country_code", "region", "income_group", "year", "value"]].rename(
        columns={"value": "life_expectancy"}
    )
    male = male[["country_name", "country_code", "year", "value"]].rename(
        columns={"value": "male_life_expectancy"}
    )
    female = female[["country_name", "country_code", "year", "value"]].rename(
        columns={"value": "female_life_expectancy"}
    )
    fertility = fertility[["country_name", "country_code", "year", "value"]].rename(
        columns={"value": "fertility_rate"}
    )
    death = death[["country_name", "country_code", "year", "value"]].rename(
        columns={"value": "death_rate"}
    )

    panel = life.merge(male, on=["country_name", "country_code", "year"], how="left")
    panel = panel.merge(female, on=["country_name", "country_code", "year"], how="left")
    panel = panel.merge(fertility, on=["country_name", "country_code", "year"], how="left")
    panel = panel.merge(death, on=["country_name", "country_code", "year"], how="left")

    panel["gender_gap"] = panel["female_life_expectancy"] - panel["male_life_expectancy"]
    panel["region"] = panel["region"].fillna("Unknown")
    panel["income_group"] = panel["income_group"].fillna("Unknown")
    panel = panel.sort_values(["country_name", "year"]).reset_index(drop=True)
    return panel


@st.cache_data
def load_eval_data() -> pd.DataFrame | None:
    if EVAL_PATH.exists():
        df = pd.read_csv(EVAL_PATH)

        rename_map = {
            "Country Name": "country_name",
            "Country Code": "country_code",
            "Year": "year",
            "Predicted": "predicted_life_expectancy",
            "Actual": "actual_life_expectancy",
        }
        df = df.rename(columns=rename_map)

        for col in ["country_name", "country_code", "year", "predicted_life_expectancy", "actual_life_expectancy"]:
            if col in df.columns:
                df[col] = df[col].astype(str) if col in {"country_name", "country_code"} else pd.to_numeric(df[col], errors="coerce")

        df["life_expectancy"] = df["actual_life_expectancy"]
        return df
    return None


panel = load_dashboard_data()
eval_df = load_eval_data()

st.title("🌐 World Bank Life Expectancy Analytics & ML Forecasting")

countries = sorted(panel["country_name"].dropna().unique())
income_groups = sorted(panel["income_group"].dropna().unique())
regions = sorted(panel["region"].dropna().unique())

# Sidebar Filters
with st.sidebar:
    st.header("📌 Dashboard Filters")

    # Smart default for country selection to keep charts clean
    default_countries = [c for c in ["United States", "China", "India", "Germany", "Central African Republic", "South Sudan"] if c in countries]
    selected_countries = st.multiselect(
        "Country Selection",
        countries,
        default=default_countries if default_countries else countries[:5],
    )
    selected_income_groups = st.multiselect(
        "Income Group",
        income_groups,
        default=income_groups,
    )
    selected_regions = st.multiselect(
        "Region",
        regions,
        default=regions,
    )

    min_year = int(panel["year"].min())
    max_year = int(panel["year"].max())
    year_range = st.slider("Year Range", min_year, max_year, (1980, max_year))

    metric = st.selectbox(
        "Primary Metric",
        ["life_expectancy", "death_rate", "fertility_rate", "gender_gap"],
        format_func=lambda x: {
            "life_expectancy": "Life Expectancy at Birth (Years)",
            "death_rate": "Death Rate (per 1,000 people)",
            "fertility_rate": "Fertility Rate (births per woman)",
            "gender_gap": "Gender Gap (Female - Male)",
        }[x],
    )

    group_by = st.radio("Group Trend By", ["Country", "Income Group", "Region"], horizontal=True)

# Apply Filters
filtered = panel[
    panel["country_name"].isin(selected_countries)
    & panel["income_group"].isin(selected_income_groups)
    & panel["region"].isin(selected_regions)
    & panel["year"].between(year_range[0], year_range[1])
].copy()

if filtered.empty:
    st.warning("⚠️ No data matches your current filters. Please adjust the sidebar selections.")
    st.stop()

# Tabs Interface
tab1, tab2, tab3 = st.tabs(["📊 Descriptive Analytics", "🤖 ML Model Forecasts", "📁 Data Explorer"])

# --- TAB 1: DESCRIPTIVE ANALYTICS ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg. Life Expectancy", f"{filtered['life_expectancy'].mean():.1f} yrs")
    with col2:
        st.metric("Avg. Fertility Rate", f"{filtered['fertility_rate'].mean():.2f}")
    with col3:
        st.metric("Avg. Death Rate", f"{filtered['death_rate'].mean():.2f}")
    with col4:
        st.metric("Avg. Gender Gap", f"{filtered['gender_gap'].mean():.2f} yrs")

    st.markdown("---")

    # Time-Series Chart
    metric_label = {
        "life_expectancy": "Life Expectancy (years)",
        "death_rate": "Death Rate (per 1,000)",
        "fertility_rate": "Fertility Rate (births per woman)",
        "gender_gap": "Gender Gap (years)",
    }[metric]

    if group_by == "Country":
        line_df = filtered[["country_name", "year", metric]].rename(columns={"country_name": "group", metric: "value"})
    else:
        grp_col = group_by.lower().replace(" ", "_")
        line_df = (
            filtered.groupby([grp_col, "year"], as_index=False)[metric]
            .mean()
            .rename(columns={grp_col: "group", metric: "value"})
        )

    fig_line = px.line(
        line_df,
        x="year",
        y="value",
        color="group",
        title=f"Trend: {metric_label} ({year_range[0]} - {year_range[1]})",
        labels={"year": "Year", "value": metric_label, "group": group_by},
    )
    fig_line.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig_line, use_container_width=True)

    # Demographic Relationship Bubble Plot (Gapminder Style)
    st.subheader("Demographic Multi-Variable Relationship")
    selected_year = st.slider("Select Snapshot Year", int(year_range[0]), int(year_range[1]), int(year_range[1]))
    
    bubble_df = filtered[filtered["year"] == selected_year].dropna(subset=["life_expectancy", "fertility_rate"])
    if not bubble_df.empty:
        fig_bubble = px.scatter(
            bubble_df,
            x="fertility_rate",
            y="life_expectancy",
            size="death_rate",
            color="income_group",
            hover_name="country_name",
            title=f"Life Expectancy vs. Fertility Rate in {selected_year} (Bubble Size = Death Rate)",
            labels={
                "fertility_rate": "Fertility Rate (Births per Woman)",
                "life_expectancy": "Life Expectancy (Years)",
                "income_group": "Income Tier",
            },
            template="plotly_white"
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

# --- TAB 2: ML MODEL FORECASTS ---
with tab2:
    st.header("Out-of-Sample Model Benchmarking (2011 - 2023)")
    
    if eval_df is not None:
        # Benchmark Metrics
        lgbm_mae, lgbm_rmse, lgbm_r2 = 0.5327, 1.2570, 0.9741
        cat_mae, cat_rmse, cat_r2 = 0.5491, 1.2483, 0.9745
        nn_mae, nn_rmse, nn_r2 = 1.0415, 1.5348, 0.9614

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("🏆 LightGBM (Selected)", f"MAE: {lgbm_mae:.4f} yrs", f"R²: {lgbm_r2:.4f}")
        with m_col2:
            st.metric("🥈 CatBoost (Runner-Up)", f"MAE: {cat_mae:.4f} yrs", f"R²: {cat_r2:.4f}")
        with m_col3:
            st.metric("🧪 PyTorch Entity NN", f"MAE: {nn_mae:.4f} yrs", f"R²: {nn_r2:.4f}")

        st.markdown("---")

        # Country Actual vs Predicted Selection
        eval_countries = sorted(eval_df["country_name"].unique())
        selected_eval_country = st.selectbox("Select Country for Model Validation Plot", eval_countries, index=0)
        
        c_eval = eval_df[eval_df["country_name"] == selected_eval_country].sort_values("year")
        c_eval_plot = c_eval[["year", "actual_life_expectancy", "predicted_life_expectancy"]].melt(
            id_vars="year",
            var_name="series",
            value_name="life_expectancy",
        )

        fig_pred = px.line(
            c_eval_plot,
            x="year",
            y="life_expectancy",
            color="series",
            labels={"life_expectancy": "Life Expectancy (Years)", "year": "Year", "series": "Series"},
            title=f"CatBoost Forecast vs. Actuals: {selected_eval_country} (2011–2023)",
            template="plotly_white",
        )
        fig_pred.update_traces(name="Actual", selector={"legendgroup": "actual_life_expectancy"})
        fig_pred.update_traces(name="CatBoost Predicted", selector={"legendgroup": "predicted_life_expectancy"})
        st.plotly_chart(fig_pred, use_container_width=True)

    else:
        st.info("ℹ️ Run `ml_forecasting.ipynb` to save `forecast_evaluation.csv` in your `data/` folder to view interactive forecast models.")

# --- TAB 3: DATA EXPLORER ---
with tab3:
    st.subheader("Filtered Panel Dataset")
    show_data = filtered[[
        "country_name", "country_code", "region", "income_group", "year",
        "life_expectancy", "male_life_expectancy", "female_life_expectancy",
        "gender_gap", "fertility_rate", "death_rate",
    ]].sort_values(["country_name", "year"]).reset_index(drop=True)
    
    st.dataframe(show_data, use_container_width=True)

    # CSV Download Button
    csv = show_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_life_expectancy_data.csv",
        mime="text/csv",
    )