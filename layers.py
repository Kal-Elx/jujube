from neuron import *

class Layer:

    def __init__(self, size : int, prev_layer_size: int, activation_func : ActivationFunction):
        self.neurons = [Neuron(activation_func=activation_func, prev_layer_size=prev_layer_size) for i in range(size)]