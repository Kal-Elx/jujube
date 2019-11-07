from neuron import *


class Layer:

    def __init__(self, size: int, prev_layer_size: int, index: int, activation_func: ActivationFunction):
        self.index: int = index
        self.neurons: List[Neuron] = [Neuron(activation_func=activation_func, prev_layer_size=prev_layer_size,
                                             index=(index, i)) for i in range(size)]
