from .weight_initializer import IWeightInitializer
import random


class RandomInitializer(IWeightInitializer):
    def init(self):
        random_number_generator = random.Random(0) # Use a fixed seed for reproducibility
        return random_number_generator.uniform(-0.5, 0.5)
