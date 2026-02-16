import pandas as pd


def test_fetch_returns_dataframe_with_expected_columns(monkeypatch):
    from app import fetch_weather

    mock_response = {
        "hourly": {
            "time": ["2026-02-15T00:00", "2026-02-15T01:00", "2026-02-15T02:00"],
            "temperature_2m": [28.0, 27.5, 27.0],
            "relative_humidity_2m": [80, 82, 85],
            "wind_speed_10m": [5.0, 4.5, 3.0],
            "cloud_cover": [20, 30, 40],
            "precipitation": [0.0, 0.0, 0.1],
            "rain": [0.0, 0.0, 0.1],
            "uv_index": [0.0, 0.0, 0.5],
            "apparent_temperature": [29.0, 28.5, 28.0],
        },
        "daily": {
            "time": ["2026-02-15"],
            "sunrise": ["2026-02-15T06:21"],
            "sunset": ["2026-02-15T18:14"],
        },
    }

    class MockResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return mock_response

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())

    df, daily = fetch_weather.__wrapped__()
    assert isinstance(df, pd.DataFrame)
    assert "temperature_2m" in df.columns
    assert "relative_humidity_2m" in df.columns
    assert "wind_speed_10m" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 3
    assert daily["sunrise"][0] == "2026-02-15T06:21"
    assert daily["sunset"][0] == "2026-02-15T18:14"
