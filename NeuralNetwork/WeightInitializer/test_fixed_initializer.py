import pytest
from .fixed_initializer import FixedInitializer
import numpy as np
import re


def test_fixed_2_3_1():
    weight_initializer = FixedInitializer(([[1, 2], [3, 4], [5, 6]], [[7, 8, 9]]))

    weights = weight_initializer.init(2, 3)
    assert np.shape(weights) == (3, 2)
    assert np.array_equal(weights, np.array([[1, 2], [3, 4], [5, 6]]))

    weights = weight_initializer.init(3, 1)
    assert np.shape(weights) == (1, 3)
    assert np.array_equal(weights, np.array([[7, 8, 9]]))

def test_wrong_shape():
    weight_initializer = FixedInitializer(([[1]]))
    with pytest.raises(ValueError, match=re.escape("Expected shape (2, 2), but got (1,)")):
        weight_initializer.init(2, 2)
