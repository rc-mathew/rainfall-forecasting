import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    import subprocess

    model = tmp_path_factory.mktemp("m") / "forecaster.joblib"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--no-backtest", "--no-mlflow",
                    "--out", str(model)], check=True, capture_output=True)
    import api.main as main
    main.MODEL_PATH = model
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("target", ["total", "max_daily"])
def test_forecast_shape_and_order(client, target):
    body = client.get("/forecast", params={"target": target}).json()
    pts = body["points"]
    assert len(pts) == 12 and pts[0]["month"] == "2020-07" and pts[-1]["month"] == "2021-06"
    for p in pts:
        assert 0 <= p["lower_80_mm"] <= p["forecast_mm"] <= p["upper_80_mm"]


def test_horizon_validation(client):
    assert client.get("/forecast", params={"horizon": 13}).status_code == 422
    assert client.get("/forecast", params={"target": "nope"}).status_code == 422
    assert len(client.get("/forecast", params={"horizon": 3}).json()["points"]) == 3


def test_december_wetter_than_february(client):
    pts = {p["month"]: p["forecast_mm"] for p in client.get("/forecast").json()["points"]}
    assert pts["2020-12"] > pts["2021-02"]
