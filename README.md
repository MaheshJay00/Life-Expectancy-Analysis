# World Bank Life Expectancy Forecasting

This project analyzes World Bank life expectancy indicators and builds a recursive forecasting model to project life expectancy from 2011 to 2023 using only information available up to 2010.

## Project basis

This project is a practical analysis of World Bank life expectancy and related demographic indicators. It focuses on cleaning and processing the available data, exploring trends across countries and income groups, building a machine learning model to forecast life expectancy, and presenting the results through an interactive dashboard.

The project uses processed World Bank indicator files stored under data/processed_data/ and produces machine learning forecast results and evaluation outputs under data/results/.

## Business and technical goal

- Load and clean multiple World Bank datasets
- Combine historical indicators into a country-year panel
- Build a forecast model using lag-based and trend-based features
- Keep the project requirement intact:
  - Train with data up to 2010
  - Forecast from 2011 to 2023
  - Do not use future actuals in recursive prediction
- Present findings in an interactive dashboard

## Model choice

The final production pipeline uses CatBoost as the main forecasting model.

This choice is based on the project results:

- CatBoost outperformed the naive 2010 baseline on MAE and RMSE
- It handled the recursive lag features well
- It remained stable across the evaluation window

## Repository structure

- `src/data_processing/` — source code for downloading and cleaning the data, downloaded data will be stored in data/raw_data/ and cleaned data will be stored in data/processed_data/
- `src/run_data_pipeline.py` — main data processing pipeline
- `src/ml_forecasting.py` — main forecasting pipeline
- `dashboard/app.py` — main Streamlit dashboard
- `notebooks/` — exploratory analysis and plotting notebooks

## Requirements

The project uses Python and the following key libraries:

- pandas
- numpy
- scikit-learn
- catboost
- plotly
- streamlit
- pytest

A project virtual environment is recommended.

## How to run

### 1) Open the project in a Python environment

From the project root:

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 1.1) Install project dependencies

```powershell
python -m pip install -r requirements.txt
```

or, if you are already in the virtual environment:

```powershell
pip install -r requirements.txt
```

### 2) Run Data extraction pipeline

```powershell
python src\run_data_pipeline.py
```

This does the following:

- downloads the data from worldback.org
- cleans the dataset
- merges metadata to contain income information
- saves them into individual csv files

### 3) Run the forecasting pipeline

```powershell
python src\ml_forecasting.py
```

This does the following:

- loads the processed country/year data
- creates recursive lag and trend features
- trains the model using data through 2010
- forecasts 2011–2023
- saves outputs to `data/results/`

### 4) Run the dashboard

```powershell
streamlit run dashboard\app.py
```

This opens the dashboard with:

- country, region, and income filters
- historical trend views
- forecast evaluation charts
- forecast comparison visuals

## Output files

The forecasting pipeline writes the following files to `data/results/`:

- `forecast_catboost.csv`
- `catboost_forecast_summary.csv`
- `catboost_forecast_horizon_metrics.csv`
- `catboost_country_metrics.csv`
- `catboost_feature_importance.csv`

These files are used by the dashboard and forecasting evaluation views.

## Conclusion
The CatBoost model achieved an overall MAE of 0.961 years, RMSE of 1.790 years, and R² of 0.948 across the 2011–2023 forecasting period. For comparison, a naive baseline that assumes each country's life expectancy remains at its 2010 level achieved an MAE of 1.828 years, RMSE of 2.780 years, and R² of 0.874. These results indicate that the CatBoost model was able to capture important historical patterns and produce substantially more accurate forecasts than simply carrying forward the 2010 value.

However, the results also demonstrate the difficulty of long-term recursive forecasting. Forecast errors can accumulate as predictions are repeatedly used as inputs, and unexpected events or structural changes after 2010 cannot be fully captured from historical life expectancy patterns alone. The model is therefore best interpreted as a data-driven forecasting approach rather than a representation of all factors that determine population health. The model cannot foresee a structural shock like COVID-19 because it is driven by historical patterns and no exogenous crisis signal is available to the training data.

Future improvements could include incorporating additional predictors such as healthcare expenditure, GDP per capita, mortality indicators, fertility rates, education, vaccination coverage, and other socioeconomic or demographic variables. Additional forecasting approaches, model ensembles, and more rigorous time-based hyperparameter tuning could also be investigated to determine whether forecast accuracy can be improved further.

## Notes

- The project requirement is to stop training at 2010 and forecast forward only from 2011.
- The raw World Bank files may contain later years, but the project workflow intentionally limits the modeling horizon to 2011–2023.
- The dashboard is intended to present the project-defined evaluation frame, not raw future data beyond the approved forecasting period.
