import pytest
from .neural_network import NeuralNetwork
from .WeightInitializer.random_initializer import RandomInitializer
from .ActivationFunction.std_logistic_function import StdLogisticFunction


def test_one_neuron():
    net = NeuralNetwork(
        neurons_per_layer=[1],
        weight_initializer=RandomInitializer(),
        activation_function=StdLogisticFunction()
    )

    # Note:
    # With an input of 0, the output should be 0.5, because the weight is 0 and sigmoid(0) = 0.5.
    # The other values are just randomly chosen.
    inputs = [0, 0.5, 1, 1e3]
    expected_outputs = [0.500, 0.542, 0.585, 1.000]

    for input_value, expected_output in zip(inputs, expected_outputs):
        outputs = net.query(inputs=[input_value])
        assert len(outputs) == 1
        assert outputs[0] == pytest.approx(expected_output, abs=1e-3)
