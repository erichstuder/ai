import pytest
from .relu import ReLU


def test_relu():
    relu = ReLU()
    assert relu.apply(-1) == 0
    assert relu.apply(0)  == 0
    assert relu.apply(1)  == 1
