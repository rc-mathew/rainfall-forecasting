"""Train the forecasters, save them for the API, and optionally log the run to MLflow.

    python scripts/train.py                      # train + backtest + MLflow (local ./mlflow.db)
    python scripts/train.py --no-backtest --no-mlflow
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainfall.data import TARGETS, load_series  # noqa: E402
from rainfall.model import DEFAULT_XGB_PARAMS, MEMBERS, RainfallForecaster, backtest  # noqa: E402

BACKTEST_ORIGINS = [pd.Timestamp(f"{y}-06-01") for y in range(2010, 2019)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "models" / "forecaster.joblib"))
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    bundle, metrics = {}, {}
    for target in TARGETS:
        series = load_series(target, args.data_dir)
        bundle[target] = RainfallForecaster().fit(series)
        if not args.no_backtest:
            for k, v in backtest(series, BACKTEST_ORIGINS).items():
                metrics[f"{target}_{k}"] = v
    info = {"members": MEMBERS, "xgb_params": DEFAULT_XGB_PARAMS,
            "data_through": str(bundle["total"].history.index[-1].date()), "backtest": metrics}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"forecasters": bundle, "info": info}, out)
    forecast_csv = out.with_name("forecast.csv")
    pd.concat({t: f.predict() for t, f in bundle.items()}, axis=1).round(2).to_csv(forecast_csv)
    print(json.dumps(info, indent=2, default=str))

    if not args.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        mlflow.set_experiment("rainfall-forecasting")
        with mlflow.start_run(run_name="ensemble"):
            mlflow.log_params({"members": ",".join(MEMBERS), "horizon": 12, **DEFAULT_XGB_PARAMS,
                               "data_through": info["data_through"]})
            mlflow.log_metrics({k: v for k, v in metrics.items()})
            mlflow.log_artifact(str(out))
            mlflow.log_artifact(str(forecast_csv))
        print("logged to MLflow: run `mlflow ui --backend-store-uri sqlite:///mlflow.db`")


if __name__ == "__main__":
    main()
