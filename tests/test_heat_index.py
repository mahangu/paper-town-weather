import pytest


def test_heat_index_below_80f_uses_steadman():
    """Below 80F threshold, Steadman simple formula is used."""
    from app import compute_heat_index
    result = compute_heat_index(25.0, 50.0)
    assert 24.5 < result < 25.5


def test_heat_index_hot_humid():
    """Hot and humid conditions should produce heat index above actual temp."""
    from app import compute_heat_index
    result = compute_heat_index(35.0, 75.0)
    assert result > 35.0


def test_heat_index_hot_dry():
    """Hot but dry conditions - heat index closer to actual temp."""
    from app import compute_heat_index
    result = compute_heat_index(35.0, 20.0)
    assert result > 30.0
    assert result < 40.0


def test_heat_index_low_humidity_adjustment():
    """RH < 13% with T between 80-112F triggers subtraction adjustment."""
    from app import compute_heat_index
    result = compute_heat_index(38.0, 10.0)
    result_no_adj = compute_heat_index(38.0, 14.0)
    assert result < result_no_adj


def test_heat_index_high_humidity_adjustment():
    """RH > 85% with T between 80-87F triggers addition adjustment."""
    from app import compute_heat_index
    result = compute_heat_index(28.0, 90.0)
    assert result > 28.0


def test_heat_index_category_comfortable():
    from app import heat_index_category
    assert heat_index_category(25.0) == "Comfortable"


def test_heat_index_category_caution():
    from app import heat_index_category
    assert heat_index_category(30.0) == "Caution"


def test_heat_index_category_extreme_caution():
    from app import heat_index_category
    assert heat_index_category(35.0) == "Extreme Caution"


def test_heat_index_category_danger():
    from app import heat_index_category
    assert heat_index_category(45.0) == "Danger"


def test_heat_index_category_extreme_danger():
    from app import heat_index_category
    assert heat_index_category(55.0) == "Extreme Danger"


def test_wet_bulb_moderate():
    from app import compute_wet_bulb
    result = compute_wet_bulb(25.0, 50.0)
    assert 16 < result < 19


def test_wet_bulb_hot_humid():
    from app import compute_wet_bulb
    result = compute_wet_bulb(35.0, 90.0)
    assert result > 30


def test_wet_bulb_hot_dry():
    from app import compute_wet_bulb
    result = compute_wet_bulb(35.0, 20.0)
    assert result < 22
