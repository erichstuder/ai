from .WeightInitializer.weight_initializer import IWeightInitializer
from .ActivationFunction.activation_function import IActivationFunction


# Note:
# - The special input layer distributing the input data is left out of the neurons_per_layer list. It is handled separately in the code.
class NeuralNetwork:
    def __init__(self,
                 neurons_per_layer,
                 weight_initializer: IWeightInitializer,
                 activation_function: IActivationFunction):
        # self.neurons_per_layer = neurons_per_layer
        self.weights = weight_initializer.init()
        self.activation_function = activation_function
    
    def query(self, inputs):
        return [self.activation_function.apply(self.weights * inputs[0])]

    # def train(self):
    #     print("Train does nothing yet, but hello anyway!")
    #     pass


