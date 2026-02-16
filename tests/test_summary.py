def test_summary_hot_humid_rain():
    from app import current_conditions_summary
    result = current_conditions_summary(35.0, 85.0, 42.0, 5.0)
    assert "Dangerously hot" in result
    assert "very humid" in result
    assert "heavy rain" in result


def test_summary_pleasant_dry():
    from app import current_conditions_summary
    result = current_conditions_summary(24.0, 50.0, 24.0, 0.0)
    assert "Pleasant" in result
    assert "dry" in result
    assert "rain" not in result


def test_summary_warm_humid_light_rain():
    from app import current_conditions_summary
    result = current_conditions_summary(30.0, 70.0, 30.0, 1.0)
    assert "Warm" in result
    assert "humid" in result
    assert "light rain" in result
