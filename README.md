# Rainfall forecasting: Changi, Singapore (PRAICP-1004)

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-EB5A0C) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white) ![statsmodels](https://img.shields.io/badge/statsmodels-4051B5) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) ![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white) ![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white) ![Last commit](https://img.shields.io/github/last-commit/rc-mathew/rainfall-forecasting)

Forecast **monthly total rainfall** and the **highest single-day rainfall in the month** for the next 12 months
(July 2020 – June 2021), from NEA data for Changi Climate Station, Jan 1982 – Jun 2020 (462 months, no gaps).

**Submission file:** `PRAICP-1004_Rainfall_Forecasting.ipynb` (already executed, outputs included).
Everything else in this repo (API, Docker, MLflow, tests) is optional engineering around the notebook.

## Result in one paragraph
Seasonality is the only strong signal (Nov–Jan wettest, Feb driest). Beyond the calendar month there is almost
no memory (lag-1 autocorrelation 0.20), so **no model significantly beats "use the historical median of that calendar
month"**. The final model is an average of Monthly median, Linear Regression, Random Forest and XGBoost; it is statistically
indistinguishable from the median baseline, and the median is an equally valid simpler choice. Forecast uncertainty is
large: 80% intervals are about 255 mm wide for monthly totals.

12-month-ahead rolling backtest, 108 out-of-sample months (MAE, mm):

| Model | Total rainfall | Highest daily |
|---|---|---|
| Seasonal naive | 86.2 | 28.8 |
| SARIMA | 68.9 | 21.8 |
| LSTM | 74.2 | 19.5 |
| Linear Reg. / Random Forest / XGBoost | 64.8 / 64.6 / 64.6 | 19.4 / 18.9 / 19.0 |
| Monthly median | 63.9 | 19.3 |
| **Ensemble (final)** | **63.7** | **19.0** |

Forecasts are median-like "typical month" values; they sum to about 1,875 mm, below the mean annual 2,118 mm.

## Layout
```
PRAICP-1004_Rainfall_Forecasting.ipynb   full analysis (EDA, tests, ARIMA/SARIMA, ML, LSTM, ensemble, forecast)
data/                                    NEA CSVs + metadata
outputs/                                 leaderboards, significance tests, holdout, forecast CSV, model card
src/rainfall/                            features + ensemble forecaster used by the API
scripts/train.py                         train, backtest, save model, log run to MLflow
api/main.py                              FastAPI service
tests/                                   pytest (features, determinism, API)
Dockerfile, requirements-api.txt         serving image
```

## Run
```bash
pip install -r requirements.txt
export PYTHONPATH=src:.
python scripts/train.py                      # trains, backtests, writes models/forecaster.joblib, logs to MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn api.main:app --port 8000             # http://localhost:8000/docs
pytest
```
```bash
curl "localhost:8000/forecast?target=total&horizon=12"      # target: total | max_daily
```

### Docker
```bash
docker build -t rainfall-api . && docker run -p 8000:8000 rainfall-api
```
The model is trained inside the image build so it matches the installed library versions.

## Notes
- All features use information at least 12 months old, so one model forecasts all 12 months directly with no leakage.
- Model order selection (AIC) uses only data before the first backtest origin; differencing is fixed from the stationarity tests.
- Data: National Environment Agency, via data.gov.sg (Singapore Open Data Licence).
