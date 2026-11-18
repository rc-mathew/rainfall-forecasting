import pandas as pd

from rainfall.data import load_series
from rainfall.model import RainfallForecaster, build_features


def test_features_use_only_old_information():
    s = load_series("total", "data")
    f = build_features(s)
    t = s.index[200]
    # lag_12 at t is exactly the value 12 months earlier, never anything newer
    assert f.loc[t, "lag_12"] == s.loc[t - pd.DateOffset(months=12)]
    # changing the future must not change today's features
    s2 = s.copy(); s2.iloc[250:] = 9999
    assert build_features(s2).loc[t].equals(f.loc[t])


def test_forecast_is_deterministic():
    s = load_series("max_daily", "data")[:"2015-06-01"]
    a = RainfallForecaster().fit(s).predict()
    b = RainfallForecaster().fit(s).predict()
    pd.testing.assert_frame_equal(a, b)
