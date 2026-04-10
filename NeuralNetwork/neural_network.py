from .WeightInitializer.weight_initializer import WeightInitializer
from .ActivationFunction.activation_function import ActivationFunction
import numpy as np


class NeuralNetwork:
    def __init__(self,
                 neurons_per_layer,
                 weight_initializer: WeightInitializer,
                 activation_function: ActivationFunction):
        self.neurons_per_layer = neurons_per_layer

        self.layer_weights = []
        self.layer_offsets = []
        for i in range(len(neurons_per_layer) - 1):
            input_size = neurons_per_layer[i]
            output_size = neurons_per_layer[i+1]
            weights, offsets = weight_initializer.init(input_size, output_size)
            self.layer_weights.append(weights)
            self.layer_offsets.append(offsets)
        
        self.activation_function = activation_function
    
    def query(self, inputs):
        outputs = np.array(inputs)
        for n in range(len(self.layer_weights)):
            weights = self.layer_weights[n]
            offset = np.array(self.layer_offsets[n]).reshape(-1, 1)
            outputs = self.activation_function.apply(np.dot(weights, outputs) + offset)
        return outputs
    
    def pretty_print_weights(self):
        np.set_printoptions(precision=3, suppress=True)
        for idx, w in enumerate(self.layer_weights):
            print(f"Layer {idx} weights (shape {np.array(w).shape}):")
            print(np.array(w))
            print("-"*40)
