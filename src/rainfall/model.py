"""12-month-ahead rainfall forecaster: average of Monthly median, Linear Regression, Random Forest, XGBoost.

All features use information at least 12 months old, so one fitted model forecasts every month of the next
year directly. Targets are modelled as log1p(rainfall). Intervals (80%) come from XGBoost quantile models.
This mirrors the notebook PRAICP-1004_Rainfall_Forecasting.ipynb.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 12
SEED = 42
DEFAULT_XGB_PARAMS = {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.02}
MEMBERS = ["Monthly median", "Linear Regression", "Random Forest", "XGBoost"]


def build_features(series: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=series.index)
    month = series.index.month
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)
    for k in (12, 24, 36):
        f[f"lag_{k}"] = series.shift(k)
    f["same_month_mean_3y"] = f[["lag_12", "lag_24", "lag_36"]].mean(axis=1)
    f["month_median_expanding"] = series.groupby(month).transform(lambda x: x.shift(1).expanding().median())
    f["rolling_12m_mean_lag12"] = series.shift(12).rolling(12).mean()
    f["rolling_12m_max_lag12"] = series.shift(12).rolling(12).max()
    return f


def future_index(series: pd.Series, horizon: int = HORIZON) -> pd.DatetimeIndex:
    return pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")


class RainfallForecaster:
    def __init__(self, xgb_params: dict | None = None):
        self.xgb_params = xgb_params or DEFAULT_XGB_PARAMS
        self.history: pd.Series | None = None
        self.models: dict = {}

    def _xgb(self, **extra):
        return xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                **self.xgb_params, **extra)

    def fit(self, series: pd.Series) -> "RainfallForecaster":
        self.history = series.copy()
        feats = build_features(series)
        valid = feats.notna().all(axis=1)
        X, y = feats[valid], np.log1p(series[valid])
        self.columns = list(X.columns)
        factories = {
            "Linear Regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
            "Random Forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=10,
                                                           max_features=0.6, random_state=SEED, n_jobs=-1),
            "XGBoost": self._xgb,
            "q10": lambda: self._xgb(objective="reg:quantileerror", quantile_alpha=0.10),
            "q90": lambda: self._xgb(objective="reg:quantileerror", quantile_alpha=0.90),
        }
        self.models = {name: make().fit(X, y) for name, make in factories.items()}
        return self

    def predict(self, horizon: int = HORIZON) -> pd.DataFrame:
        if self.history is None:
            raise RuntimeError("fit() first")
        if not 1 <= horizon <= HORIZON:
            raise ValueError(f"horizon must be between 1 and {HORIZON}")
        hist = self.history
        idx = future_index(hist)
        extended = pd.concat([hist, pd.Series(np.nan, index=idx)])
        X_future = build_features(extended).iloc[len(hist):][self.columns]

        median = hist.groupby(hist.index.month).median().loc[idx.month].values
        members = {"Monthly median": median}
        for name in ("Linear Regression", "Random Forest", "XGBoost"):
            members[name] = np.clip(np.expm1(self.models[name].predict(X_future)), 0, None)
        point = np.mean([members[m] for m in MEMBERS], axis=0)

        lower = np.clip(np.expm1(self.models["q10"].predict(X_future)), 0, None)
        upper = np.clip(np.expm1(self.models["q90"].predict(X_future)), 0, None)
        out = pd.DataFrame({"forecast": point,
                            "lower_80": np.minimum(lower, point),
                            "upper_80": np.maximum(upper, point)}, index=idx)
        return out.iloc[:horizon]


def backtest(series: pd.Series, origins: list[pd.Timestamp]) -> dict:
    """12-month-ahead rolling-origin backtest. Returns MAE of the ensemble and of the monthly-median baseline."""
    err_ens, err_med = [], []
    for origin in origins:
        train = series[:origin]
        test = series[origin + pd.offsets.MonthBegin():].iloc[:HORIZON]
        fc = RainfallForecaster().fit(train).predict()
        med = train.groupby(train.index.month).median().loc[test.index.month].values
        err_ens.append(np.abs(test.values - fc["forecast"].values[:len(test)]))
        err_med.append(np.abs(test.values - med))
    return {"mae_ensemble": float(np.concatenate(err_ens).mean()),
            "mae_monthly_median": float(np.concatenate(err_med).mean()),
            "n_months": int(sum(len(e) for e in err_ens))}
