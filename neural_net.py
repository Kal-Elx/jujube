from layer import *


class NeuralNet:

    def __init__(self, architecture: List[int], hl_act_func: ActivationFunction = ActivationFunction.SIGMOID,
                 ol_act_func: ActivationFunction = ActivationFunction.SIGMOID,
                 cost_func: CostFunction = CostFunction.QUADRATIC_COST):
        """
        :param architecture: Number of neurons in each net. [input layer, hidden layer, ..., output layer].
        :param hl_act_func: Activation function for the hidden layers.
        :param ol_act_func: Activation function for the output layer.
        """
        # Check if the network have at least input and output layer.
        assert len(architecture) >= 2, "The network needs two have at least two layers."
        # Check if all layers have at least one neuron.
        assert not any(e == 0 for e in architecture), "The network can not have layers with zero neurons."

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
        self.cost_func = get_cost_func(cost_func=cost_func)
        self.cost_func_prime = get_cost_func_prime(cost_func=cost_func)

    def exec(self, input: List[float]) -> List[float]:
        """
        Executes the neural net with the given input.
        :param input: Input to the network in the form of a list of floats of the same size as the input layer.
        :return Output for given input in the form of a list of floats of the same size as the output layer.
        """
        # Check if input is of the same size as the input layer.
        assert len(input) == self.architecture[0], "Input is of different size than the input layer."
        # Check format of input.
        assert isinstance(input, list), "Input is given in the wrong format."
        assert isinstance(input[0], float), "Input is given in the wrong format."

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
        # Test format of training data.
        assert isinstance(training_set, List), "Training data is given in the wrong format."
        assert isinstance(training_set[0], Tuple), "Training data is given in the wrong format."
        assert isinstance(training_set[0][0], List), "Training data is given in the wrong format."
        assert isinstance(training_set[0][1], List), "Training data is given in the wrong format."
        assert isinstance(training_set[0][0][0], float), "Training data is given in the wrong format."
        assert isinstance(training_set[0][1][0], float), "Training data is given in the wrong format."

        for i in range(epochs):

            # Divide the training set in mini batches.
            shuffle(training_set)
            mini_batches = [training_set[i:i + mini_batch_size] for i in range(0, len(training_set), mini_batch_size)]

            # Update weights and biases for every mini batch.
            for mini_batch in mini_batches:
                self.gradient_descent(learning_rate=learning_rate / mini_batch_size, batch=mini_batch)

            print("Completed epoch {0}".format(i + 1))

    def test(self, test_set: List[Tuple[List[float], List[float]]]) -> float:
        """
        Calculates the average error of the cost function for a given set of test data.
        :param test_set: Given test data.
        :return: Average error.
        """
        # Test format of test data.
        assert isinstance(test_set, List), "Test data is given in the wrong format."
        assert isinstance(test_set[0], Tuple), "Test data is given in the wrong format."
        assert isinstance(test_set[0][0], List), "Test data is given in the wrong format."
        assert isinstance(test_set[0][1], List), "Test data is given in the wrong format."
        assert isinstance(test_set[0][0][0], float), "Test data is given in the wrong format."
        assert isinstance(test_set[0][1][0], float), "Test data is given in the wrong format."

        # Compute the error.
        error = 0.0
        for x, y in test_set:
            error += self.cost_func(self.exec(x), y)

        return error / len(test_set)

    def gradient_descent(self, batch: List[Tuple[List[float], List[float]]], learning_rate: float) -> None:
        """
        Applies gradient descent to the weights and biases in the network.
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
            gradient_biases = [[gb + db for gb, db in zip(lgb, ldb)] for lgb, ldb in
                               zip(gradient_biases, delta_biases)]
            gradient_weights = [[[gw + dw for gw, dw in zip(ngw, ndw)] for ngw, ndw in zip(lgw, ldw)]
                                for lgw, ldw in zip(gradient_weights, delta_weights)]

        # Update weights and biases according to the obtained gradients.
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].neurons)):
                self.layers[i].neurons[j].bias -= learning_rate * gradient_biases[i][j]
                for k in range(len(self.layers[i].neurons[j].weights)):
                    self.layers[i].neurons[j].weights[k] -= learning_rate * gradient_weights[i][j][k]

    def backpropagation(self, x: List[float], y: List[float]) -> (List[float], List[List[float]]):
        """
        Calculates the gradient of the cost function with respect to the network's weights biases for the given training
        example.
        :param x: Input.
        :param y: Correct output.
        :return: The gradient of the cost function in two lists, one for biases and one for weights.
        """
        # Initialize the gradient.
        delta_biases = [[0.0] * e for e in self.architecture[1:]]
        delta_weights = [[[0.0] * f for x in range(e)] for e, f in zip(self.architecture[1:], self.architecture[:-1])]

        # Calculate the z and activations for all layers.
        self.exec(x)

        # Compute the gradient for the error in each layer.
        change = None  # Save previous value.
        for l in range(self.number_of_layers - 2, -1, -1):
            if l == self.number_of_layers - 2:
                # Equation 1
                change = np.multiply(self.cost_func_prime([n.activation for n in self.layers[l].neurons], y),
                                     [n.activation_func_prime(n.z) for n in self.layers[l].neurons])
            else:
                # Equation 2
                change = np.multiply(np.dot(transpose([n.weights for n in self.layers[l + 1].neurons]), change),
                                     [n.activation_func_prime(n.z) for n in self.layers[l].neurons])

            for j in range(self.architecture[l + 1]):
                # Equation 3
                delta_biases[l][j] = change[j]

                for k in range(self.architecture[l]):
                    # Equation 4
                    if l != 0:
                        delta_weights[l][j][k] = self.layers[l - 1].neurons[k].activation * change[j]
                    else:
                        delta_weights[l][j][k] = x[k] * change[j]  # Activation of input layer is the input

        return delta_biases, delta_weights
