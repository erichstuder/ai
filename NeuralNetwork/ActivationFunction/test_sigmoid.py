import pytest
from .std_logistic_function import StdLogisticFunction


def test_sigmoid():
    sigmoid = StdLogisticFunction()
    assert sigmoid.apply(-0.123) == pytest.approx(0.469, abs=1e-3)
    assert sigmoid.apply(0)      == pytest.approx(0.500, abs=1e-3)
    assert sigmoid.apply(0.7)    == pytest.approx(0.668, abs=1e-3)
