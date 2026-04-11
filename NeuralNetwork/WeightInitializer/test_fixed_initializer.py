import pytest
from .fixed_initializer import FixedInitializer
import numpy as np
import re


def test_weights():
    weight_initializer = FixedInitializer(([[1, 2], [3, 4], [5, 6]], [[7, 8, 9]]), [[0, 0, 0], [0]])

    weights, _ = weight_initializer.init(2, 3)
    assert np.shape(weights) == (3, 2)
    assert np.array_equal(weights, np.array([[1, 2], [3, 4], [5, 6]]))

    weights, _ = weight_initializer.init(3, 1)
    assert np.shape(weights) == (1, 3)
    assert np.array_equal(weights, np.array([[7, 8, 9]]))


def test_offsets():
    weight_initializer = FixedInitializer(([[0], [0], [0]], [[0, 0, 0]]), [[.1, .2, .3], [.4]])

    _, offsets = weight_initializer.init(1, 3)
    assert np.shape(offsets) == (3,1)
    assert np.array_equal(offsets, np.array([.1, .2, .3]).reshape(-1, 1))

    _, offsets = weight_initializer.init(3, 1)
    assert np.shape(offsets) == (1,1)
    assert np.array_equal(offsets, np.array([.4]).reshape(-1, 1))


def test_wrong_weights_shape():
    weight_initializer = FixedInitializer(weights=([[1]]), offsets=[0, 0])
    with pytest.raises(ValueError, match=re.escape("Expected weights shape (2, 2), but got (1, 1) with [[1]]")):
        weight_initializer.init(2, 2)


def test_wrong_offsets_shape():
    weight_initializer = FixedInitializer(weights=([[0],
                                                    [0]],),
                                          offsets=[1])
    with pytest.raises(ValueError, match=re.escape("Expected offsets shape (2, 1), but got (1, 1) with [[1]]")):
        weight_initializer.init(1, 2)
