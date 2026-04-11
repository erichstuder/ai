import pytest
from .random_initializer import RandomInitializer
import numpy as np


def test_random_1_1():
    weight_initializer_random = RandomInitializer()
    weights, offsets = weight_initializer_random.init(1, 1)

    assert np.shape(weights) == (1, 1)
    assert weights[0][0] == pytest.approx(0.048, abs=1e-3)

    assert np.shape(offsets) == (1, 1)
    assert offsets[0] == 0

def test_random_2_2():
    weight_initializer_random = RandomInitializer()
    weights, offsets = weight_initializer_random.init(2, 2)

    assert np.shape(weights) == (2, 2)
    assert np.all((weights >= -0.5) & (weights <= 0.5))

    assert np.shape(offsets) == (2, 1)
    assert np.all(offsets == 0)
