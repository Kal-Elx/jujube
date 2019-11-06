from layers import *

class NeuralNet:

    def __init__(self, architecture: List[int], hl_act_func: ActivationFunction = ActivationFunction.SIGMOID,
                 ol_act_func: ActivationFunction = ActivationFunction.SIGMOID):
        """
        :param architecture: Number of neurons in each net. [input layer, hidden layer, ..., output layer].
        :param hl_act_func: Activation function for the hidden layers.
        :param ol_act_func: Activation function for the output layer.
        """

        assert len(architecture) >= 2 # The network needs to have at least input and output layer.

        self.architecture = architecture

        # Initialize network.
        self.layers = []
        for i in range(1, len(architecture)-1):
            self.layers.append(Layer(size=architecture[i], prev_layer_size=architecture[i-1], activation_func=hl_act_func))
        self.layers.append(Layer(size=architecture[-1], prev_layer_size=architecture[-2], activation_func=ol_act_func))


    def exec(self, input: List[float]) -> List[float]:
        """
        Executes the neural net with the given input.
        :param input: Input to the network in the form of a list of floats of the same size as the input layer.
        :return Output for given input in the form of a list of floats of the same size as the output layer.
        """

        assert len(input) == self.architecture[0] # Input needs to be of the same size as the input layer.

        prev_layer_output = input
        for layer in self.layers:
            curr_layer_output = []
            for neuron in layer.neurons:
                curr_layer_output.append(neuron.exec(prev_layer_output))
            prev_layer_output = curr_layer_output
        return prev_layer_output
