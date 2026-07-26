"""Tests for src.pricing.black_scholes."""
import math
import pytest
from src.pricing.black_scholes import black_scholes_call_price, black_scholes_put_price

BASE_CASE = {
    "spot": 100.0,
    "strike": 100.0,
    "time_to_maturity": 1.0,
    "risk_free_rate": 0.05,
    "volatility": 0.20,
    "dividend_yield": 0.0,
}

def test_known_at_the_money_values():
    assert black_scholes_call_price(**BASE_CASE) == pytest.approx(10.4505835722, abs=1e-8)
    assert black_scholes_put_price(**BASE_CASE) == pytest.approx(5.5735260223, abs=1e-8)

def test_put_call_parity_with_dividend_yield():
    p = {"spot":112.0,"strike":105.0,"time_to_maturity":1.75,"risk_free_rate":0.043,"volatility":0.31,"dividend_yield":0.018}
    call = black_scholes_call_price(**p)
    put = black_scholes_put_price(**p)
    rhs = p["spot"]*math.exp(-p["dividend_yield"]*p["time_to_maturity"]) - p["strike"]*math.exp(-p["risk_free_rate"]*p["time_to_maturity"])
    assert call - put == pytest.approx(rhs, abs=1e-10)

@pytest.mark.parametrize(("spot","strike","expected_call","expected_put"), [(120.0,100.0,20.0,0.0),(80.0,100.0,0.0,20.0),(100.0,100.0,0.0,0.0)])
def test_expiry_equals_intrinsic(spot, strike, expected_call, expected_put):
    p = {"spot":spot,"strike":strike,"time_to_maturity":0.0,"risk_free_rate":0.05,"volatility":0.20,"dividend_yield":0.01}
    assert black_scholes_call_price(**p) == pytest.approx(expected_call)
    assert black_scholes_put_price(**p) == pytest.approx(expected_put)

def test_prices_are_non_negative():
    assert black_scholes_call_price(**BASE_CASE) >= 0.0
    assert black_scholes_put_price(**BASE_CASE) >= 0.0

def test_put_decreases_with_spot():
    assert black_scholes_put_price(**{**BASE_CASE,"spot":80.0}) > black_scholes_put_price(**{**BASE_CASE,"spot":120.0})

def test_call_increases_with_spot():
    assert black_scholes_call_price(**{**BASE_CASE,"spot":80.0}) < black_scholes_call_price(**{**BASE_CASE,"spot":120.0})

def test_values_increase_with_volatility():
    low = {**BASE_CASE,"volatility":0.10}
    high = {**BASE_CASE,"volatility":0.50}
    assert black_scholes_call_price(**low) < black_scholes_call_price(**high)
    assert black_scholes_put_price(**low) < black_scholes_put_price(**high)

@pytest.mark.parametrize(("field","value"), [("spot",0.0),("spot",-1.0),("strike",0.0),("strike",-1.0),("time_to_maturity",-0.01),("volatility",-0.01)])
def test_invalid_inputs_raise_value_error(field, value):
    p = {**BASE_CASE, field:value}
    with pytest.raises(ValueError): black_scholes_call_price(**p)
    with pytest.raises(ValueError): black_scholes_put_price(**p)
