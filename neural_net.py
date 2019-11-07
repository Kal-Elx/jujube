from layer import *


class NeuralNet:

    def __init__(self, architecture: List[int], hl_act_func: ActivationFunction = ActivationFunction.SIGMOID,
                 ol_act_func: ActivationFunction = ActivationFunction.SIGMOID):
        """
        :param architecture: Number of neurons in each net. [input layer, hidden layer, ..., output layer].
        :param hl_act_func: Activation function for the hidden layers.
        :param ol_act_func: Activation function for the output layer.
        """
        # Check if the network have at least input and output layer.
        assert len(architecture) >= 2

        # Gather information about the network´s architecture.
        self.architecture: List[int] = architecture
        self.number_of_layers: int = len(architecture)
        self.number_of_biases: int = sum(architecture[1:])

        # Initialize network.
        self.layers: List[Layer] = []
        for i in range(1, len(architecture) - 1):
            self.layers.append(Layer(size=architecture[i], prev_layer_size=architecture[i - 1], index=i,
                                     activation_func=hl_act_func))
        self.layers.append(
            Layer(size=architecture[-1], prev_layer_size=architecture[-2], index=self.number_of_layers - 1,
                  activation_func=ol_act_func))

    def exec(self, input: List[float]) -> List[float]:
        """
        Executes the neural net with the given input.
        :param input: Input to the network in the form of a list of floats of the same size as the input layer.
        :return Output for given input in the form of a list of floats of the same size as the output layer.
        """
        # Check if input is of the same size as the input layer.
        assert len(input) == self.architecture[0]

        # Calculate output of the network in a feedforward manner.
        prev_layer_output = input
        for layer in self.layers:
            curr_layer_output = []
            for neuron in layer.neurons:
                curr_layer_output.append(neuron.exec(prev_layer_output))
            prev_layer_output = curr_layer_output

        # Return output from the output layer.
        return prev_layer_output

    def train(self, training_set: List[Tuple[List[float], List[float]]], epochs: int, mini_batch_size: int,
              learning_rate: float) -> None:
        """
        Train the network on the given training data using stochastic gradient descent.
        :param training_set: Given training data.
        :param epochs: Number of epochs (iterations) to apply stochastic gradient descent.
        :param mini_batch_size: Number of training examples in each mini batch.
        :param learning_rate: The learning rate for stochastic gradient descent.
        """
        for i in range(epochs):

            # Divide the training set in mini batches.
            shuffle(training_set)
            mini_batches = [training_set[i:i + mini_batch_size] for i in range(0, len(training_set), mini_batch_size)]

            # Update weights and biases for every mini batch.
            for mini_batch in mini_batches:
                self.gradient_descent(learning_rate=learning_rate, batch=mini_batch)
                return

    def gradient_descent(self, batch: List[Tuple[List[float], List[float]]], learning_rate: float) -> None:
        """
        Apply gradient descent to the weights and biases in the network.
        :param batch: Given training data.
        :param learning_rate: The learning rate for stochastic gradient descent.
        """
        # Initialize the gradients.
        gradient_biases = [[0.0] * e for e in self.architecture[1:]]
        gradient_weights = [[[0.0] * f for x in range(e)] for e, f in
                            zip(self.architecture[1:], self.architecture[:-1])]

        # Calculate the gradients.
        for x, y in batch:
            delta_biases, delta_weights = self.backpropagation(x=x, y=y)
            gradient_biases = [[gb + learning_rate * db for gb, db in zip(lgb, ldb)] for lgb, ldb in
                               zip(gradient_biases, delta_biases)]
            gradient_weights = [[[gw + learning_rate * dw for gw, dw in zip(ngw, ndw)] for ngw, ndw in zip(lgw, ldw)]
                                for lgw, ldw in zip(gradient_weights, delta_weights)]

        # Update weights and biases according to the obtained gradients.
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].neurons)):
                self.layers[i].neurons[j].bias += gradient_biases[i][j]
                for k in range(len(self.layers[i].neurons[j].weights)):
                    self.layers[i].neurons[j].weights[k] += gradient_weights[i][j][k]

    def backpropagation(self, x: List, y: List) -> (List[float], List[List[float]]):
        # dummy implementation
        a = [[-0.2] * e for e in self.architecture[1:]]
        b = [[[0.1] * f for x in range(e)] for e, f in zip(self.architecture[1:], self.architecture[:-1])]
        return a, b

    def test(self, test_set: List[Tuple[List[float], List[float]]]) -> float:
        pass
