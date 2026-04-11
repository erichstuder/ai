from .weight_initializer import WeightInitializer
import numpy as np


class FixedInitializer(WeightInitializer):
    def __init__(self, weights, offsets):
        """
        Args:
            weights (list or array-like): List of weight matrices for each layer. Each element should be a 2D array with shape (output_size, input_size).
            offsets (list or array-like): List of offset vectors for each layer. Each element should be a 1D or 2D array with shape (output_size,) or (output_size, 1).
        """
        if not isinstance(weights, (list, tuple)):
            weights = [weights]
        if not isinstance(offsets, (list, tuple)):
            offsets = [offsets]
        self.weights = weights
        self.offsets = offsets
        self.current_layer = 0

    def init(self, input_size, output_size):
        weights = np.array(self.weights[self.current_layer])
        if weights.ndim <= 1:
            weights = weights.reshape(-1, 1)
        expected_shape = (output_size, input_size)
        if np.shape(weights) != expected_shape:
            raise ValueError(f"Expected weights shape {expected_shape}, but got {np.shape(weights)} with {weights}")
        
        offsets = np.array(self.offsets[self.current_layer]).reshape(-1, 1)
        expected_shape = (output_size, 1)
        if np.shape(offsets) != expected_shape:
            raise ValueError(f"Expected offsets shape {expected_shape}, but got {np.shape(offsets)} with {offsets}")

        self.current_layer += 1
        return (weights, offsets)
