from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_data"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

LIFE_EXPECTANCY_FILE = (
    PROCESSED_DIR / "processed_life_expectancy_total.csv"
)

FORECAST_OUTPUT_FILE = (
    RESULTS_DIR / "forecast_catboost.csv"
)

SUMMARY_OUTPUT_FILE = (
    RESULTS_DIR / "catboost_forecast_summary.csv"
)

HORIZON_OUTPUT_FILE = (
    RESULTS_DIR / "catboost_forecast_horizon_metrics.csv"
)

COUNTRY_OUTPUT_FILE = (
    RESULTS_DIR / "catboost_country_metrics.csv"
)

IMPORTANCE_OUTPUT_FILE = (
    RESULTS_DIR / "catboost_feature_importance.csv"
)


# ============================================================
# FORECAST CONFIGURATION
# ============================================================

FORECAST_ORIGIN = 2010
FORECAST_START = 2011
FORECAST_END = 2023

LAGS = [1, 2, 3, 4, 5]

MIN_LEVEL_HISTORY = max(LAGS) + 1

RANDOM_STATE = 42


FEATURE_COLUMNS = [
    "lag_1_life",
    "lag_2_life",
    "lag_3_life",
    "lag_4_life",
    "lag_5_life",
    "life_change_1",
    "life_change_3",
    "year",
]

VALIDATION_YEARS = 2


def safe_improvement(baseline: float, current: float) -> float:
    """Return percentage improvement, guarded against zero baseline values."""
    if baseline == 0:
        return 0.0
    return ((baseline - current) / baseline) * 100.0


def create_validation_split(
    df: pd.DataFrame,
    validation_years: int = VALIDATION_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a country-level panel by year into train and validation sets."""
    if validation_years <= 0:
        raise ValueError("validation_years must be positive")

    years = sorted(df["year"].dropna().unique())
    if len(years) <= validation_years:
        raise ValueError("Not enough years to create a validation split")

    validation_end = years[-1]
    validation_start = validation_end - validation_years + 1

    train_df = df[df["year"] < validation_start].copy()
    val_df = df[df["year"].between(validation_start, validation_end)].copy()

    return train_df, val_df


# ============================================================
# LOAD DATA
# ============================================================

def load_life_expectancy_data() -> pd.DataFrame:
    """
    Load processed total life expectancy data.

    Expected columns:
        country_name
        country_code
        year
        value
    """

    if not LIFE_EXPECTANCY_FILE.exists():
        raise FileNotFoundError(
            f"Life expectancy file not found:\n"
            f"{LIFE_EXPECTANCY_FILE}"
        )

    df = pd.read_csv(LIFE_EXPECTANCY_FILE)

    required_columns = {
        "country_name",
        "country_code",
        "year",
        "value",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = df[
        [
            "country_name",
            "country_code",
            "year",
            "value",
        ]
    ].copy()

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "country_code",
            "year",
        ]
    )

    df["year"] = df["year"].astype(int)

    df = df.rename(
        columns={
            "value": "life_expectancy",
        }
    )

    # Check for duplicate country/year combinations.
    duplicates = df.duplicated(
        subset=[
            "country_code",
            "year",
        ],
        keep=False,
    )

    if duplicates.any():
        duplicate_rows = df.loc[
            duplicates,
            [
                "country_name",
                "country_code",
                "year",
            ],
        ]

        raise ValueError(
            "Duplicate country/year observations found:\n"
            f"{duplicate_rows.head(20).to_string(index=False)}"
        )

    df = df.sort_values(
        [
            "country_code",
            "year",
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_training_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create historical life-expectancy change features.

    The target is:

        delta(t) = life_expectancy(t)
                   - life_expectancy(t-1)

    Lag features contain historical deltas only.
    """

    result = df.copy()

    result = result.sort_values(
        [
            "country_code",
            "year",
        ]
    ).reset_index(drop=True)

    grouped = result.groupby(
        "country_code",
        group_keys=False,
    )

    result["delta"] = grouped[
        "life_expectancy"
    ].diff(1)

    delta_grouped = result.groupby(
        "country_code",
        group_keys=False,
    )

    for lag in LAGS:
        result[f"lag_{lag}_life"] = (
            delta_grouped["delta"].shift(lag)
        )

    result["life_change_1"] = (
        result["lag_1_life"]
        - result["lag_2_life"]
    )

    result["life_change_3"] = (
        result["lag_1_life"]
        - result["lag_4_life"]
    )

    return result


# ============================================================
# MODEL
# ============================================================

def build_model() -> CatBoostRegressor:
    """
    Build the final CatBoost model.

    These parameters match the benchmarked CatBoost model.
    """

    return CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        random_seed=RANDOM_STATE,
        verbose=0,
    )


# ============================================================
# RECURSIVE FORECAST FEATURES
# ============================================================

def make_recursive_features(
    history: list[float],
    year: int,
) -> dict[str, float]:
    """
    Create features for one recursive forecast step.

    history contains actual observations through 2010
    followed by previous model predictions.
    """

    if len(history) < MIN_LEVEL_HISTORY:
        raise ValueError(
            "Not enough historical observations to "
            f"create lag features. Need at least "
            f"{MIN_LEVEL_HISTORY} observations."
        )

    features: dict[str, float] = {}

    for lag in LAGS:
        features[f"lag_{lag}_life"] = (
            history[-lag]
            - history[-lag - 1]
        )

    features["life_change_1"] = (
        features["lag_1_life"]
        - features["lag_2_life"]
    )

    features["life_change_3"] = (
        features["lag_1_life"]
        - features["lag_4_life"]
    )

    features["year"] = year

    return features


# ============================================================
# FORECAST ONE COUNTRY
# ============================================================

def forecast_country(
    model: CatBoostRegressor,
    country_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recursively forecast one country/series from 2011 to 2023.

    IMPORTANT:
        Actual future observations are NOT fed back into history.

        2010 actual
            ↓
        predict 2011
            ↓
        use predicted 2011
            ↓
        predict 2012
            ↓
        ...
            ↓
        predict 2023
    """

    country_df = country_df.sort_values(
        "year"
    ).copy()

    historical = country_df[
        country_df["year"] <= FORECAST_ORIGIN
    ].copy()

    historical = historical.dropna(
        subset=["life_expectancy"]
    )

    if historical.empty:
        return pd.DataFrame()

    if len(historical) < MIN_LEVEL_HISTORY:
        return pd.DataFrame()

    # We require an actual observation at the
    # forecast origin.
    if historical["year"].max() != FORECAST_ORIGIN:
        return pd.DataFrame()

    history = historical[
        "life_expectancy"
    ].tolist()

    # Naive baseline:
    # hold the 2010 value constant.
    baseline_prediction = history[-1]

    forecasts = []

    for year in range(
        FORECAST_START,
        FORECAST_END + 1,
    ):

        feature_dict = make_recursive_features(
            history=history,
            year=year,
        )

        X_future = pd.DataFrame(
            [feature_dict],
            columns=FEATURE_COLUMNS,
        )

        predicted_delta = float(
            model.predict(X_future)[0]
        )

        prediction = (
            history[-1]
            + predicted_delta
        )

        # Retrieve actual only for evaluation.
        # It is NOT added to history.
        actual_row = country_df[
            country_df["year"] == year
        ]

        actual = np.nan

        if not actual_row.empty:
            actual = actual_row[
                "life_expectancy"
            ].iloc[0]

        forecasts.append(
            {
                "country_name": country_df[
                    "country_name"
                ].iloc[0],

                "country_code": country_df[
                    "country_code"
                ].iloc[0],

                "year": year,

                "actual_life_expectancy": actual,

                "predicted_life_expectancy": prediction,

                "baseline_prediction": baseline_prediction,
            }
        )

        # CRITICAL:
        # Append the prediction, NOT the actual.
        history.append(prediction)

    return pd.DataFrame(forecasts)


# ============================================================
# FORECAST ALL COUNTRIES
# ============================================================

def generate_forecasts(
    model: CatBoostRegressor,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate recursive forecasts for all eligible
    country/series.
    """

    forecast_frames = []

    skipped_series = []

    for country_code, country_df in df.groupby(
        "country_code"
    ):

        forecast = forecast_country(
            model=model,
            country_df=country_df,
        )

        if forecast.empty:
            skipped_series.append(
                country_code
            )
            continue

        forecast_frames.append(
            forecast
        )

    if not forecast_frames:
        raise ValueError(
            "No country/series could be forecast."
        )

    forecast_df = pd.concat(
        forecast_frames,
        ignore_index=True,
    )

    forecast_df["error"] = (
        forecast_df[
            "predicted_life_expectancy"
        ]
        - forecast_df[
            "actual_life_expectancy"
        ]
    )

    forecast_df["absolute_error"] = (
        forecast_df["error"].abs()
    )

    forecast_df["baseline_absolute_error"] = (
        forecast_df[
            "baseline_prediction"
        ]
        - forecast_df[
            "actual_life_expectancy"
        ]
    ).abs()

    print(
        f"Successfully forecast "
        f"{forecast_df['country_code'].nunique()} "
        "series."
    )

    if skipped_series:
        print(
            f"Skipped {len(skipped_series)} series "
            "because they did not meet the "
            "forecasting requirements."
        )

    return forecast_df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """
    Calculate MAE, RMSE and R².
    """

    valid = (
        actual.notna()
        & predicted.notna()
    )

    actual_valid = actual.loc[valid]
    predicted_valid = predicted.loc[valid]

    if len(actual_valid) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
        }

    return {
        "MAE": mean_absolute_error(
            actual_valid,
            predicted_valid,
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                actual_valid,
                predicted_valid,
            )
        ),

        "R2": r2_score(
            actual_valid,
            predicted_valid,
        ),
    }


# ============================================================
# OVERALL EVALUATION
# ============================================================

def evaluate_overall(
    forecast_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Evaluate CatBoost against the naive 2010 baseline.
    """

    catboost_metrics = calculate_metrics(
        forecast_df[
            "actual_life_expectancy"
        ],

        forecast_df[
            "predicted_life_expectancy"
        ],
    )

    baseline_metrics = calculate_metrics(
        forecast_df[
            "actual_life_expectancy"
        ],

        forecast_df[
            "baseline_prediction"
        ],
    )

    evaluation = pd.DataFrame(
        [
            {
                "Model": "CatBoost",
                **catboost_metrics,
            },
            {
                "Model": "Naive 2010 Baseline",
                **baseline_metrics,
            },
        ]
    )

    return (
        evaluation,
        catboost_metrics,
        baseline_metrics,
    )


# ============================================================
# HORIZON METRICS
# ============================================================

def calculate_horizon_metrics(
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate metrics separately for each forecast year.
    """

    rows = []

    for year, year_df in forecast_df.groupby(
        "year"
    ):

        catboost = calculate_metrics(
            year_df[
                "actual_life_expectancy"
            ],

            year_df[
                "predicted_life_expectancy"
            ],
        )

        baseline = calculate_metrics(
            year_df[
                "actual_life_expectancy"
            ],

            year_df[
                "baseline_prediction"
            ],
        )

        rows.append(
            {
                "Year": int(year),

                "Horizon": int(
                    year - FORECAST_ORIGIN
                ),

                "CatBoost_MAE": catboost["MAE"],

                "CatBoost_RMSE": catboost["RMSE"],

                "CatBoost_R2": catboost["R2"],

                "Baseline_MAE": baseline["MAE"],

                "Baseline_RMSE": baseline["RMSE"],

                "Baseline_R2": baseline["R2"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        "Year"
    )


# ============================================================
# COUNTRY METRICS
# ============================================================

def calculate_country_metrics(
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate metrics for each country/series.
    """

    rows = []

    for country_code, country_df in forecast_df.groupby(
        "country_code"
    ):

        catboost = calculate_metrics(
            country_df[
                "actual_life_expectancy"
            ],

            country_df[
                "predicted_life_expectancy"
            ],
        )

        baseline = calculate_metrics(
            country_df[
                "actual_life_expectancy"
            ],

            country_df[
                "baseline_prediction"
            ],
        )

        rows.append(
            {
                "Country Name": country_df[
                    "country_name"
                ].iloc[0],

                "Country Code": country_code,

                "CatBoost MAE": catboost["MAE"],

                "CatBoost RMSE": catboost["RMSE"],

                "CatBoost R2": catboost["R2"],

                "Baseline MAE": baseline["MAE"],

                "Baseline RMSE": baseline["RMSE"],

                "Baseline R2": baseline["R2"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        "CatBoost MAE"
    )


# ============================================================
# SAVE DASHBOARD FORECAST
# ============================================================

def save_dashboard_forecast(
    forecast_df: pd.DataFrame,
) -> None:
    """
    Save forecast in the exact format expected
    by the Streamlit dashboard.
    """

    dashboard_forecast = forecast_df[
        [
            "country_name",
            "country_code",
            "year",
            "predicted_life_expectancy",
            "actual_life_expectancy",
        ]
    ].rename(
        columns={
            "country_name": "Country Name",
            "country_code": "Country Code",
            "year": "Year",
            "predicted_life_expectancy": "Predicted",
            "actual_life_expectancy": "Actual",
        }
    )

    dashboard_forecast.to_csv(
        FORECAST_OUTPUT_FILE,
        index=False,
    )


# ============================================================
# MAIN FORECASTING PIPELINE
# ============================================================

def run_forecasting_pipeline() -> None:
    """
    Run the complete CatBoost forecasting pipeline.
    """

    print("=" * 70)
    print("LIFE EXPECTANCY CATBOOST FORECASTING")
    print("=" * 70)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    print("\n[1/7] Loading life expectancy data...")

    df = load_life_expectancy_data()

    print(
        f"Loaded {len(df):,} observations."
    )

    print(
        f"Countries/series: "
        f"{df['country_code'].nunique():,}"
    )

    print(
        f"Year range: "
        f"{df['year'].min()}–{df['year'].max()}"
    )

    # --------------------------------------------------------
    # 2. Create features
    # --------------------------------------------------------

    print("\n[2/7] Creating training features...")

    featured_df = create_training_features(
        df
    )

    # --------------------------------------------------------
    # 3. Training data
    # --------------------------------------------------------

    print(
        "\n[3/7] Preparing training data "
        f"through {FORECAST_ORIGIN}..."
    )

    training_df = featured_df[
        featured_df["year"]
        <= FORECAST_ORIGIN
    ].copy()

    training_df = training_df.dropna(
        subset=FEATURE_COLUMNS + ["delta"]
    )

    X_train = training_df[
        FEATURE_COLUMNS
    ]

    y_train = training_df[
        "delta"
    ]

    print(
        f"Training observations: "
        f"{len(training_df):,}"
    )

    print(
        f"Training features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    validation_train_df, validation_df = create_validation_split(
        featured_df,
        validation_years=VALIDATION_YEARS,
    )

    validation_train_df = validation_train_df.dropna(subset=FEATURE_COLUMNS + ["delta"])
    validation_df = validation_df.dropna(subset=["life_expectancy"])

    if validation_train_df.empty or validation_df.empty:
        logger.warning("Validation split produced an empty train or validation set; continuing with full-year training.")
    else:
        logger.info(
            "Validation split created: train years %s to %s, validation years %s to %s",
            int(validation_train_df["year"].min()),
            int(validation_train_df["year"].max()),
            int(validation_df["year"].min()),
            int(validation_df["year"].max()),
        )

    # --------------------------------------------------------
    # 4. Train CatBoost
    # --------------------------------------------------------

    print("\n[4/7] Training CatBoost...")

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    print("CatBoost training completed.")

    # --------------------------------------------------------
    # 5. Generate recursive forecasts
    # --------------------------------------------------------

    print(
        "\n[5/7] Generating recursive forecasts "
        f"for {FORECAST_START}–{FORECAST_END}..."
    )

    forecast_df = generate_forecasts(
        model=model,
        df=df,
    )

    # --------------------------------------------------------
    # 6. Evaluate
    # --------------------------------------------------------

    print("\n[6/7] Evaluating forecasts...")

    (
        evaluation,
        catboost_metrics,
        baseline_metrics,
    ) = evaluate_overall(
        forecast_df
    )

    horizon_metrics = (
        calculate_horizon_metrics(
            forecast_df
        )
    )

    country_metrics = (
        calculate_country_metrics(
            forecast_df
        )
    )

    feature_importance = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": model.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    # --------------------------------------------------------
    # 7. Save outputs
    # --------------------------------------------------------

    print("\n[7/7] Saving results...")

    save_dashboard_forecast(
        forecast_df
    )

    horizon_metrics.to_csv(
        HORIZON_OUTPUT_FILE,
        index=False,
    )

    country_metrics.to_csv(
        COUNTRY_OUTPUT_FILE,
        index=False,
    )

    feature_importance.to_csv(
        IMPORTANCE_OUTPUT_FILE,
        index=False,
    )

    catboost_mae = catboost_metrics["MAE"]
    baseline_mae = baseline_metrics["MAE"]

    catboost_rmse = catboost_metrics["RMSE"]
    baseline_rmse = baseline_metrics["RMSE"]

    mae_improvement = safe_improvement(baseline_mae, catboost_mae)
    rmse_improvement = safe_improvement(baseline_rmse, catboost_rmse)

    summary = pd.DataFrame(
        [
            {
                "Model": "CatBoost",

                "Forecast Origin": (
                    FORECAST_ORIGIN
                ),

                "Forecast Start": (
                    FORECAST_START
                ),

                "Forecast End": (
                    FORECAST_END
                ),

                "MAE": catboost_metrics["MAE"],

                "RMSE": catboost_metrics["RMSE"],

                "R2": catboost_metrics["R2"],

                "Series Forecast": (
                    forecast_df[
                        "country_code"
                    ].nunique()
                ),

                "MAE Improvement vs Baseline (%)": (
                    mae_improvement
                ),

                "RMSE Improvement vs Baseline (%)": (
                    rmse_improvement
                ),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Forecast origin : {FORECAST_ORIGIN}"
    )

    print(
        f"Forecast period : "
        f"{FORECAST_START}–{FORECAST_END}"
    )

    print(
        f"Series forecast : "
        f"{forecast_df['country_code'].nunique():,}"
    )

    print()
    print("CatBoost:")
    print(
        f"  MAE  : "
        f"{catboost_metrics['MAE']:.3f} years"
    )

    print(
        f"  RMSE : "
        f"{catboost_metrics['RMSE']:.3f} years"
    )

    print(
        f"  R²   : "
        f"{catboost_metrics['R2']:.3f}"
    )

    print()
    print("Naive 2010 baseline:")
    print(
        f"  MAE  : "
        f"{baseline_metrics['MAE']:.3f} years"
    )

    print(
        f"  RMSE : "
        f"{baseline_metrics['RMSE']:.3f} years"
    )

    print(
        f"  R²   : "
        f"{baseline_metrics['R2']:.3f}"
    )

    print()
    print(
        f"MAE improvement vs baseline: "
        f"{mae_improvement:.1f}%"
    )

    print(
        f"RMSE improvement vs baseline: "
        f"{rmse_improvement:.1f}%"
    )

    print()
    print("Saved files:")

    print(
        f"  {FORECAST_OUTPUT_FILE}"
    )

    print(
        f"  {SUMMARY_OUTPUT_FILE}"
    )

    print(
        f"  {HORIZON_OUTPUT_FILE}"
    )

    print(
        f"  {COUNTRY_OUTPUT_FILE}"
    )

    print(
        f"  {IMPORTANCE_OUTPUT_FILE}"
    )

    print("=" * 70)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_forecasting_pipeline()
