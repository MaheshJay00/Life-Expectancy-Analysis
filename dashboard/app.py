from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="World Bank Life Expectancy Analytics",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

# app.py is inside:
# project/
# └── dashboard/
#     └── app.py
#
# Therefore parents[1] points to the project root.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_FORECAST_END = 2023

DATA_DIR = PROJECT_ROOT / "data" / "processed_data"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


LIFE_EXPECTANCY_FILE = DATA_DIR / "processed_life_expectancy_total.csv"
MALE_FILE = DATA_DIR / "processed_life_expectancy_male.csv"
FEMALE_FILE = DATA_DIR / "processed_life_expectancy_female.csv"
FERTILITY_FILE = DATA_DIR / "processed_fertility_rate.csv"
DEATH_FILE = DATA_DIR / "processed_death_rate.csv"

FORECAST_FILE = RESULTS_DIR / "forecast_catboost.csv"
SUMMARY_FILE = RESULTS_DIR / "catboost_forecast_summary.csv"
HORIZON_FILE = RESULTS_DIR / "catboost_forecast_horizon_metrics.csv"
COUNTRY_METRICS_FILE = RESULTS_DIR / "catboost_country_metrics.csv"
FEATURE_IMPORTANCE_FILE = RESULTS_DIR / "catboost_feature_importance.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_metric(value: float, decimals: int = 3) -> str:
    """Format numeric dashboard metrics safely."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def check_required_files() -> list[str]:
    """Return a list of missing files."""
    required_files = [
        LIFE_EXPECTANCY_FILE,
        MALE_FILE,
        FEMALE_FILE,
        FERTILITY_FILE,
        DEATH_FILE,
    ]

    return [str(path) for path in required_files if not path.exists()]


# ============================================================
# LOAD DESCRIPTIVE DATA
# ============================================================

@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    """Load and combine all processed World Bank datasets."""

    life = pd.read_csv(LIFE_EXPECTANCY_FILE)
    male = pd.read_csv(MALE_FILE)
    female = pd.read_csv(FEMALE_FILE)
    fertility = pd.read_csv(FERTILITY_FILE)
    death = pd.read_csv(DEATH_FILE)

    datasets = [life, male, female, fertility, death]

    for df in datasets:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # --------------------------------------------------------
    # Life expectancy
    # --------------------------------------------------------

    life = life[
        [
            "country_name",
            "country_code",
            "region",
            "income_group",
            "year",
            "value",
        ]
    ].rename(
        columns={
            "value": "life_expectancy",
        }
    )

    # --------------------------------------------------------
    # Male life expectancy
    # --------------------------------------------------------

    male = male[
        [
            "country_name",
            "country_code",
            "year",
            "value",
        ]
    ].rename(
        columns={
            "value": "male_life_expectancy",
        }
    )

    # --------------------------------------------------------
    # Female life expectancy
    # --------------------------------------------------------

    female = female[
        [
            "country_name",
            "country_code",
            "year",
            "value",
        ]
    ].rename(
        columns={
            "value": "female_life_expectancy",
        }
    )

    # --------------------------------------------------------
    # Fertility
    # --------------------------------------------------------

    fertility = fertility[
        [
            "country_name",
            "country_code",
            "year",
            "value",
        ]
    ].rename(
        columns={
            "value": "fertility_rate",
        }
    )

    # --------------------------------------------------------
    # Death rate
    # --------------------------------------------------------

    death = death[
        [
            "country_name",
            "country_code",
            "year",
            "value",
        ]
    ].rename(
        columns={
            "value": "death_rate",
        }
    )

    # --------------------------------------------------------
    # Merge datasets
    # --------------------------------------------------------

    panel = life.merge(
        male,
        on=["country_name", "country_code", "year"],
        how="left",
    )

    panel = panel.merge(
        female,
        on=["country_name", "country_code", "year"],
        how="left",
    )

    panel = panel.merge(
        fertility,
        on=["country_name", "country_code", "year"],
        how="left",
    )

    panel = panel.merge(
        death,
        on=["country_name", "country_code", "year"],
        how="left",
    )

    # --------------------------------------------------------
    # Calculate gender gap
    # --------------------------------------------------------

    panel["gender_gap"] = (
        panel["female_life_expectancy"]
        - panel["male_life_expectancy"]
    )

    panel["region"] = panel["region"].fillna("Unknown")
    panel["income_group"] = panel["income_group"].fillna("Unknown")

    panel = panel[
        panel["year"] <= PROJECT_FORECAST_END
    ].copy()

    panel = (
        panel
        .sort_values(["country_name", "year"])
        .reset_index(drop=True)
    )

    return panel


# ============================================================
# LOAD CATBOOST FORECAST DATA
# ============================================================

@st.cache_data
def load_forecast_data() -> pd.DataFrame | None:
    """Load CatBoost forecast results."""

    if not FORECAST_FILE.exists():
        return None

    df = pd.read_csv(FORECAST_FILE)

    rename_map = {
        "Country Name": "country_name",
        "Country Code": "country_code",
        "Year": "year",
        "Predicted": "predicted_life_expectancy",
        "Actual": "actual_life_expectancy",
    }

    df = df.rename(columns=rename_map)

    required_columns = [
        "country_name",
        "country_code",
        "year",
        "predicted_life_expectancy",
        "actual_life_expectancy",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        st.error(
            "The CatBoost forecast file is missing required columns: "
            + ", ".join(missing)
        )
        return None

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["predicted_life_expectancy"] = pd.to_numeric(
        df["predicted_life_expectancy"],
        errors="coerce",
    )

    df["actual_life_expectancy"] = pd.to_numeric(
        df["actual_life_expectancy"],
        errors="coerce",
    )

    return df


# ============================================================
# LOAD CATBOOST SUMMARY
# ============================================================

@st.cache_data
def load_summary_data() -> pd.DataFrame | None:
    """Load overall CatBoost evaluation summary if available."""

    if not SUMMARY_FILE.exists():
        return None

    try:
        return pd.read_csv(SUMMARY_FILE)
    except Exception:
        return None


# ============================================================
# LOAD HORIZON METRICS
# ============================================================

@st.cache_data
def load_horizon_metrics() -> pd.DataFrame | None:
    """Load forecast-horizon evaluation metrics."""

    if not HORIZON_FILE.exists():
        return None

    try:
        return pd.read_csv(HORIZON_FILE)
    except Exception:
        return None


# ============================================================
# LOAD COUNTRY METRICS
# ============================================================

@st.cache_data
def load_country_metrics() -> pd.DataFrame | None:
    """Load country-level CatBoost metrics."""

    if not COUNTRY_METRICS_FILE.exists():
        return None

    try:
        return pd.read_csv(COUNTRY_METRICS_FILE)
    except Exception:
        return None


# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

@st.cache_data
def load_feature_importance() -> pd.DataFrame | None:
    """Load CatBoost feature importance."""

    if not FEATURE_IMPORTANCE_FILE.exists():
        return None

    try:
        return pd.read_csv(FEATURE_IMPORTANCE_FILE)
    except Exception:
        return None


# ============================================================
# LOAD ALL DATA
# ============================================================

missing_files = check_required_files()

if missing_files:

    st.error("❌ Required dashboard data files are missing.")

    st.markdown("The following files could not be found:")

    for file_path in missing_files:
        st.code(file_path)

    st.info(
        "Please run the data-processing pipeline first and make sure "
        "the processed files exist under data/processed_data/."
    )

    st.stop()


panel = load_dashboard_data()
forecast_df = load_forecast_data()
summary_df = load_summary_data()
horizon_df = load_horizon_metrics()
country_metrics_df = load_country_metrics()
feature_importance_df = load_feature_importance()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🌐 World Bank Life Expectancy Analytics & ML Forecasting")

st.caption(
    "Historical World Bank indicators, demographic comparisons, "
    "and out-of-sample CatBoost life-expectancy forecasting."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

countries = sorted(
    panel["country_name"]
    .dropna()
    .unique()
)

income_groups = sorted(
    panel["income_group"]
    .dropna()
    .unique()
)

regions = sorted(
    panel["region"]
    .dropna()
    .unique()
)


with st.sidebar:

    st.header("📌 Dashboard Filters")

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    default_countries = [
        country
        for country in [
            "United States",
            "China",
            "India",
            "Germany",
            "Central African Republic",
            "South Sudan",
        ]
        if country in countries
    ]

    selected_countries = st.multiselect(
        "Country Selection",
        countries,
        default=(
            default_countries
            if default_countries
            else countries[:5]
        ),
    )

    # --------------------------------------------------------
    # Income group
    # --------------------------------------------------------

    selected_income_groups = st.multiselect(
        "Income Group",
        income_groups,
        default=income_groups,
    )

    # --------------------------------------------------------
    # Region
    # --------------------------------------------------------

    selected_regions = st.multiselect(
        "Region",
        regions,
        default=regions,
    )

    # --------------------------------------------------------
    # Year range
    # --------------------------------------------------------

    min_year = int(panel["year"].min())
    max_year = min(int(panel["year"].max()), PROJECT_FORECAST_END)

    default_start_year = max(
        min_year,
        1980,
    )

    year_range = st.slider(
        "Year Range",
        min_year,
        max_year,
        (
            default_start_year,
            max_year,
        ),
    )

    # --------------------------------------------------------
    # Main metric
    # --------------------------------------------------------

    metric = st.selectbox(
        "Primary Metric",
        [
            "life_expectancy",
            "death_rate",
            "fertility_rate",
            "gender_gap",
        ],
        format_func=lambda x: {
            "life_expectancy":
                "Life Expectancy at Birth (Years)",
            "death_rate":
                "Death Rate (per 1,000 people)",
            "fertility_rate":
                "Fertility Rate (births per woman)",
            "gender_gap":
                "Gender Gap (Female - Male)",
        }[x],
    )

    # --------------------------------------------------------
    # Grouping
    # --------------------------------------------------------

    group_by = st.radio(
        "Group Trend By",
        [
            "Country",
            "Income Group",
            "Region",
        ],
        horizontal=True,
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = panel[
    panel["country_name"].isin(selected_countries)
    & panel["income_group"].isin(selected_income_groups)
    & panel["region"].isin(selected_regions)
    & panel["year"].between(
        year_range[0],
        year_range[1],
    )
].copy()


if filtered.empty:

    st.warning(
        "⚠️ No data matches your current filters. "
        "Please adjust the sidebar selections."
    )

    st.stop()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Descriptive Analytics",
        "🤖 ML Model Forecasts",
        "📁 Data Explorer",
    ]
)


# ============================================================
# TAB 1 — DESCRIPTIVE ANALYTICS
# ============================================================

with tab1:

    st.header("Descriptive Analytics")

    # --------------------------------------------------------
    # KPI cards
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_life = filtered["life_expectancy"].mean()

        st.metric(
            "Avg. Life Expectancy",
            (
                f"{avg_life:.1f} yrs"
                if not pd.isna(avg_life)
                else "N/A"
            ),
        )

    with col2:
        avg_fertility = filtered["fertility_rate"].mean()

        st.metric(
            "Avg. Fertility Rate",
            (
                f"{avg_fertility:.2f}"
                if not pd.isna(avg_fertility)
                else "N/A"
            ),
        )

    with col3:
        avg_death = filtered["death_rate"].mean()

        st.metric(
            "Avg. Death Rate",
            (
                f"{avg_death:.2f}"
                if not pd.isna(avg_death)
                else "N/A"
            ),
        )

    with col4:
        avg_gap = filtered["gender_gap"].mean()

        st.metric(
            "Avg. Female − Male Gap",
            (
                f"{avg_gap:.2f} yrs"
                if not pd.isna(avg_gap)
                else "N/A"
            ),
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Trend chart
    # --------------------------------------------------------

    metric_label = {
        "life_expectancy":
            "Life Expectancy (years)",
        "death_rate":
            "Death Rate (per 1,000)",
        "fertility_rate":
            "Fertility Rate (births per woman)",
        "gender_gap":
            "Gender Gap (years)",
    }[metric]

    if group_by == "Country":

        line_df = filtered[
            [
                "country_name",
                "year",
                metric,
            ]
        ].rename(
            columns={
                "country_name": "group",
                metric: "value",
            }
        )

    else:

        group_column = {
            "Income Group": "income_group",
            "Region": "region",
        }[group_by]

        line_df = (
            filtered
            .groupby(
                [
                    group_column,
                    "year",
                ],
                as_index=False,
            )[metric]
            .mean()
            .rename(
                columns={
                    group_column: "group",
                    metric: "value",
                }
            )
        )

    fig_line = px.line(
        line_df,
        x="year",
        y="value",
        color="group",
        title=(
            f"Trend: {metric_label} "
            f"({year_range[0]}–{year_range[1]})"
        ),
        labels={
            "year": "Year",
            "value": metric_label,
            "group": group_by,
        },
    )

    fig_line.update_layout(
        template="plotly_white",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_line,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Snapshot year
    # --------------------------------------------------------

    st.subheader(
        "Demographic Multi-Variable Relationship"
    )

    selected_year = st.slider(
        "Select Snapshot Year",
        int(year_range[0]),
        int(year_range[1]),
        int(year_range[1]),
    )

    bubble_df = filtered[
        filtered["year"] == selected_year
    ].dropna(
        subset=[
            "life_expectancy",
            "fertility_rate",
        ]
    )

    if not bubble_df.empty:

        fig_bubble = px.scatter(
            bubble_df,
            x="fertility_rate",
            y="life_expectancy",
            size="death_rate",
            color="income_group",
            hover_name="country_name",
            title=(
                f"Life Expectancy vs. Fertility Rate "
                f"in {selected_year}"
            ),
            labels={
                "fertility_rate":
                    "Fertility Rate (Births per Woman)",
                "life_expectancy":
                    "Life Expectancy (Years)",
                "income_group":
                    "Income Group",
                "death_rate":
                    "Death Rate",
            },
            template="plotly_white",
        )

        st.plotly_chart(
            fig_bubble,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Male vs Female comparison
    # --------------------------------------------------------

    st.subheader(
        "Male vs. Female Life Expectancy"
    )

    gender_df = filtered.dropna(
        subset=[
            "male_life_expectancy",
            "female_life_expectancy",
        ]
    ).copy()

    if not gender_df.empty:

        if group_by == "Country":

            gender_plot = gender_df[
                [
                    "country_name",
                    "year",
                    "male_life_expectancy",
                    "female_life_expectancy",
                ]
            ].copy()

            gender_plot = gender_plot.melt(
                id_vars=[
                    "country_name",
                    "year",
                ],
                value_vars=[
                    "male_life_expectancy",
                    "female_life_expectancy",
                ],
                var_name="sex",
                value_name="life_expectancy",
            )

            gender_plot["sex"] = gender_plot[
                "sex"
            ].map(
                {
                    "male_life_expectancy": "Male",
                    "female_life_expectancy": "Female",
                }
            )

            fig_gender = px.line(
                gender_plot,
                x="year",
                y="life_expectancy",
                color="sex",
                facet_col="country_name",
                facet_col_wrap=3,
                title="Male vs. Female Life Expectancy",
                labels={
                    "life_expectancy":
                        "Life Expectancy (Years)",
                    "year": "Year",
                    "sex": "Sex",
                },
                template="plotly_white",
            )

        else:

            group_column = {
                "Income Group": "income_group",
                "Region": "region",
            }[group_by]

            gender_grouped = (
                gender_df
                .groupby(
                    [
                        group_column,
                        "year",
                    ],
                    as_index=False,
                )[
                    [
                        "male_life_expectancy",
                        "female_life_expectancy",
                    ]
                ]
                .mean()
            )

            gender_plot = gender_grouped.melt(
                id_vars=[
                    group_column,
                    "year",
                ],
                value_vars=[
                    "male_life_expectancy",
                    "female_life_expectancy",
                ],
                var_name="sex",
                value_name="life_expectancy",
            )

            gender_plot["sex"] = gender_plot[
                "sex"
            ].map(
                {
                    "male_life_expectancy": "Male",
                    "female_life_expectancy": "Female",
                }
            )

            fig_gender = px.line(
                gender_plot,
                x="year",
                y="life_expectancy",
                color="sex",
                facet_col=group_column,
                facet_col_wrap=3,
                title=(
                    f"Male vs. Female Life Expectancy "
                    f"by {group_by}"
                ),
                labels={
                    "life_expectancy":
                        "Life Expectancy (Years)",
                    "year": "Year",
                    "sex": "Sex",
                },
                template="plotly_white",
            )

        st.plotly_chart(
            fig_gender,
            use_container_width=True,
        )


# ============================================================
# TAB 2 — ML MODEL FORECASTS
# ============================================================

with tab2:

    st.header(
        "🤖 CatBoost Life Expectancy Forecast"
    )

    st.markdown(
        """
        The model was trained using information available up to and
        including **2010** and evaluated on the period **2011–2023**.

        Future actual life-expectancy values are not used to generate
        subsequent recursive predictions.
        """
    )

    if forecast_df is None:

        st.warning(
            "⚠️ CatBoost forecast results were not found."
        )

        st.info(
            "Run the following command from the project root first:"
        )

        st.code(
            "python src\\ml_forecasting.py",
            language="powershell",
        )

    else:

        # ----------------------------------------------------
        # Calculate metrics directly from forecast file
        # ----------------------------------------------------

        evaluation_df = forecast_df.dropna(
            subset=[
                "actual_life_expectancy",
                "predicted_life_expectancy",
            ]
        ).copy()

        errors = (
            evaluation_df["predicted_life_expectancy"]
            - evaluation_df["actual_life_expectancy"]
        )

        mae = errors.abs().mean()

        rmse = np.sqrt(
            np.mean(errors ** 2)
        )

        actual_mean = (
            evaluation_df["actual_life_expectancy"]
            .mean()
        )

        ss_res = np.sum(errors ** 2)

        ss_tot = np.sum(
            (
                evaluation_df["actual_life_expectancy"]
                - actual_mean
            ) ** 2
        )

        r2 = (
            1 - ss_res / ss_tot
            if ss_tot != 0
            else np.nan
        )

        # ----------------------------------------------------
        # Naive baseline
        # ----------------------------------------------------

        baseline_df = evaluation_df.copy()

        baseline_2010 = (
            panel[
                panel["year"] == 2010
            ][
                [
                    "country_name",
                    "country_code",
                    "life_expectancy",
                ]
            ]
            .drop_duplicates(
                subset=[
                    "country_name",
                    "country_code",
                ]
            )
        )

        baseline_df = baseline_df.merge(
            baseline_2010,
            on=[
                "country_name",
                "country_code",
            ],
            how="left",
        )

        baseline_df = baseline_df.dropna(
            subset=["life_expectancy"]
        )

        baseline_errors = (
            baseline_df["life_expectancy"]
            - baseline_df["actual_life_expectancy"]
        )

        baseline_mae = baseline_errors.abs().mean()

        baseline_rmse = np.sqrt(
            np.mean(baseline_errors ** 2)
        )

        baseline_actual_mean = (
            baseline_df["actual_life_expectancy"]
            .mean()
        )

        baseline_ss_res = np.sum(
            baseline_errors ** 2
        )

        baseline_ss_tot = np.sum(
            (
                baseline_df["actual_life_expectancy"]
                - baseline_actual_mean
            ) ** 2
        )

        baseline_r2 = (
            1 - baseline_ss_res / baseline_ss_tot
            if baseline_ss_tot != 0
            else np.nan
        )

        # ----------------------------------------------------
        # Model information
        # ----------------------------------------------------

        forecast_start = int(
            evaluation_df["year"].min()
        )

        forecast_end = int(
            evaluation_df["year"].max()
        )

        number_series = evaluation_df[
            [
                "country_name",
                "country_code",
            ]
        ].drop_duplicates().shape[0]

        # ----------------------------------------------------
        # KPI cards
        # ----------------------------------------------------

        st.subheader(
            f"Model Performance: {forecast_start}–{forecast_end}"
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:

            st.metric(
                "CatBoost MAE",
                f"{mae:.3f} years",
            )

        with metric_col2:

            st.metric(
                "CatBoost RMSE",
                f"{rmse:.3f} years",
            )

        with metric_col3:

            st.metric(
                "CatBoost R²",
                f"{r2:.3f}",
            )

        st.caption(
            f"Evaluated across {number_series} country/series "
            f"with available forecast observations."
        )

        st.markdown("---")

        # ----------------------------------------------------
        # Baseline comparison
        # ----------------------------------------------------

        st.subheader(
            "Comparison with Naive 2010 Baseline"
        )

        baseline_col1, baseline_col2, baseline_col3 = st.columns(3)

        with baseline_col1:

            st.metric(
                "Naive MAE",
                f"{baseline_mae:.3f} years",
            )

        with baseline_col2:

            st.metric(
                "Naive RMSE",
                f"{baseline_rmse:.3f} years",
            )

        with baseline_col3:

            st.metric(
                "Naive R²",
                f"{baseline_r2:.3f}",
            )

        mae_improvement = (
            (baseline_mae - mae)
            / baseline_mae
            * 100
            if baseline_mae != 0
            else np.nan
        )

        rmse_improvement = (
            (baseline_rmse - rmse)
            / baseline_rmse
            * 100
            if baseline_rmse != 0
            else np.nan
        )

        st.info(
            f"Relative to the naive 2010 baseline, the CatBoost "
            f"forecast has a {mae_improvement:.1f}% lower MAE and "
            f"a {rmse_improvement:.1f}% lower RMSE in this evaluation."
        )

        # ----------------------------------------------------
        # Actual vs predicted country plot
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "Actual vs. Predicted Life Expectancy"
        )

        eval_countries = sorted(
            evaluation_df["country_name"]
            .dropna()
            .unique()
        )

        selected_eval_country = st.selectbox(
            "Select Country / Series",
            eval_countries,
        )

        country_eval = (
            evaluation_df[
                evaluation_df["country_name"]
                == selected_eval_country
            ]
            .sort_values("year")
            .copy()
        )

        country_plot = country_eval[
            [
                "year",
                "actual_life_expectancy",
                "predicted_life_expectancy",
            ]
        ].melt(
            id_vars="year",
            var_name="series",
            value_name="life_expectancy",
        )

        country_plot["series"] = country_plot[
            "series"
        ].map(
            {
                "actual_life_expectancy": "Actual",
                "predicted_life_expectancy":
                    "CatBoost Predicted",
            }
        )

        fig_pred = px.line(
            country_plot,
            x="year",
            y="life_expectancy",
            color="series",
            markers=True,
            title=(
                f"CatBoost Forecast vs. Actual: "
                f"{selected_eval_country}"
            ),
            labels={
                "year": "Year",
                "life_expectancy":
                    "Life Expectancy (Years)",
                "series": "Series",
            },
            template="plotly_white",
        )

        fig_pred.update_layout(
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_pred,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Selected-country error
        # ----------------------------------------------------

        country_errors = (
            country_eval["predicted_life_expectancy"]
            - country_eval["actual_life_expectancy"]
        )

        country_mae = country_errors.abs().mean()

        country_rmse = np.sqrt(
            np.mean(country_errors ** 2)
        )

        error_col1, error_col2 = st.columns(2)

        with error_col1:

            st.metric(
                f"{selected_eval_country} MAE",
                f"{country_mae:.3f} years",
            )

        with error_col2:

            st.metric(
                f"{selected_eval_country} RMSE",
                f"{country_rmse:.3f} years",
            )

        # ----------------------------------------------------
        # Forecast horizon metrics
        # ----------------------------------------------------

        if horizon_df is not None:

            st.markdown("---")

            st.subheader(
                "Forecast Accuracy by Horizon"
            )

            horizon_display = horizon_df.copy()

            # Normalize common column names.
            horizon_display = horizon_display.rename(
                columns={
                    "Horizon":
                        "Forecast Horizon",
                    "horizon":
                        "Forecast Horizon",
                    "MAE":
                        "MAE",
                    "RMSE":
                        "RMSE",
                    "R^2":
                        "R²",
                    "R2":
                        "R²",
                }
            )

            st.dataframe(
                horizon_display,
                use_container_width=True,
                hide_index=True,
            )

            # Try to identify numeric horizon and MAE columns
            horizon_column = next(
                (
                    col
                    for col in [
                        "Forecast Horizon",
                        "Horizon",
                        "horizon",
                    ]
                    if col in horizon_display.columns
                ),
                None,
            )

            mae_column = next(
                (
                    col
                    for col in ["MAE", "mae"]
                    if col in horizon_display.columns
                ),
                None,
            )

            if (
                horizon_column is not None
                and mae_column is not None
            ):

                fig_horizon = px.line(
                    horizon_display,
                    x=horizon_column,
                    y=mae_column,
                    markers=True,
                    title="MAE by Forecast Horizon",
                    labels={
                        horizon_column:
                            "Forecast Horizon",
                        mae_column:
                            "MAE (Years)",
                    },
                    template="plotly_white",
                )

                st.plotly_chart(
                    fig_horizon,
                    use_container_width=True,
                )

        # ----------------------------------------------------
        # Country-level metrics
        # ----------------------------------------------------

        if country_metrics_df is not None:

            st.markdown("---")

            st.subheader(
                "Country-Level Forecast Performance"
            )

            st.dataframe(
                country_metrics_df,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        if feature_importance_df is not None:

            st.markdown("---")

            st.subheader(
                "CatBoost Feature Importance"
            )

            importance_df = feature_importance_df.copy()

            # Detect likely feature/importance columns.
            feature_column = next(
                (
                    col
                    for col in [
                        "Feature",
                        "feature",
                        "Feature Name",
                        "feature_name",
                    ]
                    if col in importance_df.columns
                ),
                None,
            )

            importance_column = next(
                (
                    col
                    for col in [
                        "Importance",
                        "importance",
                        "Feature Importance",
                    ]
                    if col in importance_df.columns
                ),
                None,
            )

            if (
                feature_column is not None
                and importance_column is not None
            ):

                importance_df[
                    importance_column
                ] = pd.to_numeric(
                    importance_df[
                        importance_column
                    ],
                    errors="coerce",
                )

                importance_df = (
                    importance_df
                    .dropna(
                        subset=[
                            importance_column
                        ]
                    )
                    .sort_values(
                        importance_column,
                        ascending=False,
                    )
                )

                fig_importance = px.bar(
                    importance_df,
                    x=importance_column,
                    y=feature_column,
                    orientation="h",
                    title="CatBoost Feature Importance",
                    labels={
                        importance_column:
                            "Importance",
                        feature_column:
                            "Feature",
                    },
                    template="plotly_white",
                )

                fig_importance.update_layout(
                    yaxis={
                        "categoryorder": "total ascending"
                    }
                )

                st.plotly_chart(
                    fig_importance,
                    use_container_width=True,
                )

            else:

                st.dataframe(
                    importance_df,
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# TAB 3 — DATA EXPLORER
# ============================================================

with tab3:

    st.header("📁 Data Explorer")

    show_data = filtered[
        [
            "country_name",
            "country_code",
            "region",
            "income_group",
            "year",
            "life_expectancy",
            "male_life_expectancy",
            "female_life_expectancy",
            "gender_gap",
            "fertility_rate",
            "death_rate",
        ]
    ].sort_values(
        [
            "country_name",
            "year",
        ]
    ).reset_index(
        drop=True
    )

    st.dataframe(
        show_data,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # CSV download
    # --------------------------------------------------------

    csv_data = show_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_life_expectancy_data.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "World Bank Life Expectancy Analytics | "
    "Forecast origin: 2010 | "
    "Forecast period: 2011–2023 | "
    "Model: CatBoost"
)




