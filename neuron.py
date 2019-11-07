from utils import *


class Neuron:

    def __init__(self, activation_func: ActivationFunction, prev_layer_size: int, index: (int, int)):
        """
        :param activation_func: Activation function.
        :param prev_layer_size: Size of previous layer.
        """
        self.index: (int, int) = index
        self.bias: float = np.random.randn(1, 1)[0][0]
        self.weights: List[float] = [x[0] for x in np.random.randn(prev_layer_size, 1)]
        self.activation_func: Callable = self.get_activation_func(activation_func=activation_func)

    def exec(self, input: List[float]) -> float:
        return self.activation_func(np.dot(self.weights, input) + self.bias)

    def get_activation_func(self, activation_func: ActivationFunction) -> Callable:
        if activation_func == ActivationFunction.SIGMOID:
            return sigmoid
        elif activation_func == ActivationFunction.TANH:
            return np.tanh
        elif activation_func == ActivationFunction.RELU:
            return relu
        elif activation_func == ActivationFunction.LINEAR:
            return linear
