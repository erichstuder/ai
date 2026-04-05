import pytest
from .random_initializer import RandomInitializer


def test_random():
    weight_initializer_random = RandomInitializer()
    assert weight_initializer_random.init() == pytest.approx(0.344, abs=1e-3)
