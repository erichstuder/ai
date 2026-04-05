import pytest
from .sigmoid import Sigmoid


def test_sigmoid():
    sigmoid = Sigmoid()
    assert sigmoid.apply(-0.123) == pytest.approx(0.469, abs=1e-3)
    assert sigmoid.apply(0)      == pytest.approx(0.500, abs=1e-3)
    assert sigmoid.apply(0.7)    == pytest.approx(0.668, abs=1e-3)
