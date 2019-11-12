from utils import *


class Neuron:

    def __init__(self, activation_func: ActivationFunction, prev_layer_size: int, index: (int, int)):
        """
        :param activation_func: The activation function to use for the neuron.
        :param prev_layer_size: Number of neurons in the previous layer.
        :param index: Index in network (layer, neuron).
        """
        self.index: (int, int) = index
        self.bias: float = np.random.randn(1, 1)[0][0]
        self.weights: List[float] = [x[0] for x in np.random.randn(prev_layer_size, 1)]
        self.activation_func: Callable = get_activation_func(activation_func=activation_func)
        self.activation_func_prime: Callable = get_activation_func_prime(activation_func=activation_func)
        self.activation: float = 0
        self.z: float = 0

    def exec(self, input: List[float]) -> float:
        self.z = np.dot(self.weights, input) + self.bias
        self.activation = self.activation_func(self.z)
        return self.activation
