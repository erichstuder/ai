import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from .neural_network import NeuralNetwork
from .WeightInitializer.weight_initializer import WeightInitializer
from .WeightInitializer.random_initializer import RandomInitializer
from .ActivationFunction.activation_function import ActivationFunction
from .ActivationFunction.std_logistic_function import StdLogisticFunction


def test_one_neuron():
    net = NeuralNetwork(
        neurons_per_layer=[1, 1],
        weight_initializer=RandomInitializer(),
        activation_function=StdLogisticFunction()
    )

    # Note:
    # With an input of 0, the output should be 0.5, because the weight is 0 and sigmoid(0) = 0.5.
    # The other values are just randomly chosen.
    inputs = [0, 0.5, 1, 1e3]
    expected_outputs = [0.500, 0.506, 0.512, 1.000]

    for input_value, expected_output in zip(inputs, expected_outputs):
        outputs = net.query(inputs=[input_value])
        assert len(outputs) == 1
        assert outputs[0] == pytest.approx(expected_output, abs=1e-3)


def test_1_2_1_architecture_weights(mocker):
    weight_initializer_mock = mocker.Mock(WeightInitializer)
    weight_initializer_mock.init.side_effect = lambda input_size, output_size: (np.ones((output_size, input_size)), np.zeros(output_size))

    activation_function_mock = mocker.Mock(ActivationFunction)
    activation_function_mock.apply.side_effect = lambda x: x

    net = NeuralNetwork(
        neurons_per_layer=[1, 2, 1],
        weight_initializer=weight_initializer_mock,
        activation_function=activation_function_mock
    )

    assert net.query(1) == [2]


def test_1_2_1_architecture_offsets(mocker):
    weight_initializer_mock = mocker.Mock(WeightInitializer)
    weight_initializer_mock.init.side_effect = lambda input_size, output_size: (np.ones((output_size, input_size)), np.ones(output_size))

    activation_function_mock = mocker.Mock(ActivationFunction)
    activation_function_mock.apply.side_effect = lambda x: x

    net = NeuralNetwork(
        neurons_per_layer=[1, 2, 1],
        weight_initializer=weight_initializer_mock,
        activation_function=activation_function_mock
    )

    assert net.query(1) == [5]
