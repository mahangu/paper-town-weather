import pandas as pd


def test_find_outdoor_windows_basic():
    from app import find_outdoor_windows

    times = pd.date_range("2026-02-15 06:00", periods=12, freq="h", tz="Asia/Colombo")
    heat_indices = [25, 24, 26, 25, 24, 33, 35, 34, 33, 26, 25, 24]
    wet_bulb = [22, 21, 23, 22, 21, 28, 30, 29, 28, 23, 22, 21]
    precip = [0] * 12
    df = pd.DataFrame({"heat_index": heat_indices, "wet_bulb": wet_bulb, "precipitation": precip}, index=times)
    windows = find_outdoor_windows(df)
    assert len(windows) == 2
    assert windows[0]["hours"] == 5
    assert windows[1]["hours"] == 3


def test_find_outdoor_windows_no_good_times():
    from app import find_outdoor_windows

    times = pd.date_range("2026-02-15 06:00", periods=6, freq="h", tz="Asia/Colombo")
    heat_indices = [33, 35, 34, 33, 32, 34]
    wet_bulb = [28, 30, 29, 28, 27, 29]
    precip = [0] * 6
    df = pd.DataFrame({"heat_index": heat_indices, "wet_bulb": wet_bulb, "precipitation": precip}, index=times)
    windows = find_outdoor_windows(df)
    assert len(windows) == 0


def test_find_outdoor_windows_returns_all():
    from app import find_outdoor_windows

    times = pd.date_range("2026-02-15 06:00", periods=20, freq="h", tz="Asia/Colombo")
    heat_indices = [25, 25, 33, 25, 25, 33, 25, 25, 33, 25, 25, 33, 35, 35, 35, 35, 35, 35, 35, 35]
    wet_bulb = [22, 22, 28, 22, 22, 28, 22, 22, 28, 22, 22, 28, 30, 30, 30, 30, 30, 30, 30, 30]
    precip = [0] * 20
    df = pd.DataFrame({"heat_index": heat_indices, "wet_bulb": wet_bulb, "precipitation": precip}, index=times)
    windows = find_outdoor_windows(df)
    assert len(windows) == 4


def test_find_outdoor_windows_sorted_chronologically():
    from app import find_outdoor_windows

    times = pd.date_range("2026-02-15 06:00", periods=10, freq="h", tz="Asia/Colombo")
    heat_indices = [25, 25, 33, 33, 25, 25, 25, 25, 33, 33]
    wet_bulb = [22, 22, 28, 28, 22, 22, 22, 22, 28, 28]
    precip = [0] * 10
    df = pd.DataFrame({"heat_index": heat_indices, "wet_bulb": wet_bulb, "precipitation": precip}, index=times)
    windows = find_outdoor_windows(df)
    assert windows[0]["start"] < windows[1]["start"]


def test_find_outdoor_windows_excludes_heavy_rain():
    from app import find_outdoor_windows

    times = pd.date_range("2026-02-15 06:00", periods=6, freq="h", tz="Asia/Colombo")
    heat_indices = [25, 25, 25, 25, 25, 25]
    wet_bulb = [22, 22, 22, 22, 22, 22]
    precip = [0, 0, 3.0, 0, 0, 0]
    df = pd.DataFrame({"heat_index": heat_indices, "wet_bulb": wet_bulb, "precipitation": precip}, index=times)
    windows = find_outdoor_windows(df)
    assert len(windows) == 2
    assert windows[0]["hours"] == 2
    assert windows[1]["hours"] == 3


def test_find_outdoor_windows_allows_light_rain():
    from app import find_outdoor_windows

    times = pd.date_range("2026-02-15 06:00", periods=4, freq="h", tz="Asia/Colombo")
    heat_indices = [25, 25, 25, 25]
    wet_bulb = [22, 22, 22, 22]
    precip = [0, 1.5, 0.5, 0]
    df = pd.DataFrame({"heat_index": heat_indices, "wet_bulb": wet_bulb, "precipitation": precip}, index=times)
    windows = find_outdoor_windows(df)
    assert len(windows) == 1
    assert windows[0]["hours"] == 4


def test_find_outdoor_windows_excludes_high_wet_bulb():
    from app import find_outdoor_windows

    times = pd.date_range("2026-02-15 06:00", periods=6, freq="h", tz="Asia/Colombo")
    heat_indices = [25, 25, 25, 25, 25, 25]
    wet_bulb = [22, 22, 25, 22, 22, 22]
    precip = [0] * 6
    df = pd.DataFrame({"heat_index": heat_indices, "wet_bulb": wet_bulb, "precipitation": precip}, index=times)
    windows = find_outdoor_windows(df)
    assert len(windows) == 2
    assert windows[0]["hours"] == 2
    assert windows[1]["hours"] == 3


def test_time_of_day_label():
    from app import time_of_day_label

    assert time_of_day_label(2) == "Night"
    assert time_of_day_label(5) == "Early Morning"
    assert time_of_day_label(9) == "Morning"
    assert time_of_day_label(12) == "Midday"
    assert time_of_day_label(15) == "Afternoon"
    assert time_of_day_label(18) == "Evening"
    assert time_of_day_label(22) == "Night"
