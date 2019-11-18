from utils import *


class NeuralNet:

    def __init__(self, architecture: List[int], hl_act_func: ActivationFunction = ActivationFunction.SIGMOID,
                 ol_act_func: ActivationFunction = ActivationFunction.SIGMOID,
                 cost_func: CostFunction = CostFunction.QUADRATIC_COST,
                 regularization_technique: RegularizationTechnique = None):
        """
        :param architecture: Number of neurons in each net. [input layer, hidden layer, ..., output layer].
        :param hl_act_func: Activation function for the hidden layers.
        :param ol_act_func: Activation function for the output layer.
        :param cost_func: Cost function for the network.
        """
        # Test given architecture.
        assert len(architecture) >= 2, "The network needs two have at least two layers."
        assert not any(e == 0 for e in architecture), "The network can not have layers with zero neurons."

        # Issue warnings.
        if hl_act_func == ActivationFunction.LINEAR:
            warnings.warn("Linear function is not recommended as activation function in hidden layers.")
        if cost_func == CostFunction.CROSS_ENTROPY and ol_act_func == ActivationFunction.LINEAR:
            warnings.warn("Cross-entropy is not recommended as cost function when activation function for the output "
                          "layer is linear.")

        # Save information about the network´s architecture.
        self.architecture = architecture

        # Initialize network.
        self.weights = [np.random.normal(0.0, 1 / np.sqrt(b), (a, b)) for a, b in zip(self.architecture[1:],
                                                                                  self.architecture[:-1])]
        self.biases = [np.random.randn(a, 1) for a in self.architecture[1:]]
        self.zs = [np.empty((a, 1)) for a in self.architecture]
        self.activations = [np.empty((a, 1)) for a in self.architecture]

        # Initialize functions.
        self.hl_act_func, self.hl_act_func_prime = get_activation_func(activation_func=hl_act_func)
        self.ol_act_func, self.ol_act_func_prime = get_activation_func(activation_func=ol_act_func)
        self.cost_func, self.cost_func_prime = get_cost_func(cost_func=cost_func)
        if cost_func == CostFunction.CROSS_ENTROPY:
            self.ol_act_func_prime = no_ol_act_func_prime
        self.regularization_func = get_regularization_technique(regularization_technique=regularization_technique)

        # Initialize momentum variables.
        self.vws = [np.zeros((a, b)) for a, b in zip(self.architecture[1:], self.architecture[:-1])]
        self.vbs = [np.zeros((a, 1)) for a in self.architecture[1:]]

    def save(self, file: str) -> None:
        """
        Save the neural net to file.
        :param file: The desired path and name of the file.
        """
        filehandler = open(file, 'wb')
        pickle.dump(self.__dict__, filehandler)

    @staticmethod
    def load(file: str) -> 'NeuralNet':
        """
        Load a neural net from file.
        :param file: The path and name of the file.
        :return: An instance of the neural net.
        """
        filehandler = open(file, 'rb')
        nn = NeuralNet.__new__(NeuralNet)
        nn.__dict__.update(pickle.load(filehandler))
        return nn

    def exec(self, input: np.ndarray) -> np.ndarray:
        """
        Executes the neural net with the given input.
        :param input: Vertical numpy array of the same size as the number of neurons in the input layer.
        :return Output for given input in the form of a vertical numpy array of the same size as the number of neurons
                in the output layer.
        """
        # Test format of input.
        assert len(input) == self.architecture[0], "Input is of different size than the input layer."
        assert isinstance(input, np.ndarray), "Input is given in the wrong format."

        # Calculate the activations in the network in a feedforward manner.
        self.activations[0] = input
        for i, (w, b) in enumerate(iterable=zip(self.weights, self.biases), start=1):
            if i == len(self.architecture): # Output layer
                act_func = self.ol_act_func
            else: # Hidden layer
                act_func = self.hl_act_func
            self.zs[i] = np.matmul(w, self.activations[i-1]) + b
            self.activations[i] = act_func(self.zs[i])

        # Return activations of the output layer.
        return self.activations[-1]

    def train(self, training_set: List[Tuple[np.ndarray, np.ndarray]], epochs: int, mini_batch_size: int,
              learning_rate: float, regularization: float = 0.0, momentum_coefficient: float = 0.0,
              print_progress: bool = False) -> None:
        """
        Train the network on the given training data using stochastic gradient descent.
        :param training_set: Given training data.
        :param epochs: Number of epochs (iterations) to apply stochastic gradient descent.
        :param mini_batch_size: Number of training examples in each mini batch.
        :param learning_rate: The learning rate for stochastic gradient descent.
        :param regularization: Regularization parameter. Use 0.0 for no regularization.
        :param momentum_coefficient: Momentum co-efficient for the momentum technique. Use 0.0 for no momentum.
        :param print_progress: Print the progress of the learning process.
        """
        # Test format of training data.
        assert isinstance(training_set, List), "Training data is given in the wrong format."
        assert isinstance(training_set[0], Tuple), "Training data is given in the wrong format."
        assert isinstance(training_set[0][0], np.ndarray), "Training data is given in the wrong format."
        assert isinstance(training_set[0][1], np.ndarray), "Training data is given in the wrong format."

        # Save current time for measuring the time of the training process.
        start_time = time.time()

        # Perform stochastic gradient descent.
        for i in range(epochs):

            # Divide the training set into mini batches.
            shuffle(training_set)
            mini_batches = [training_set[i:i + mini_batch_size] for i in range(0, len(training_set), mini_batch_size)]

            # Update weights and biases for every mini batch.
            for j, mini_batch in enumerate(iterable=mini_batches, start=1):
                self.gradient_descent(batch=mini_batch, learning_rate=learning_rate / mini_batch_size,
                                      regularization=regularization / len(mini_batches),
                                      momentum_coefficient=momentum_coefficient)

                if print_progress and j % 100 == 0:
                    print("Epoch: {0}/{1}, Mini batch: {2}/{3}".format(i + 1, epochs, j, len(mini_batches)))

        if print_progress:
            print("\nTraining time: {0} min {1} sec".format(round((time.time()-start_time) // 60), round(time.time()-start_time) % 60))

    def test(self, test_set: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Computes the average cost of the given test examples.
        :param test_set: Given test data.
        :return: Average cost of test examples.
        """
        # Test format of test data.
        assert isinstance(test_set, List), "Test data is given in the wrong format."
        assert isinstance(test_set[0], Tuple), "Test data is given in the wrong format."
        assert isinstance(test_set[0][0], np.ndarray), "Test data is given in the wrong format."
        assert isinstance(test_set[0][1], np.ndarray), "Test data is given in the wrong format."

        # Compute the average cost.
        total_cost = 0.0
        for x, y in test_set:
            total_cost += self.cost_func(self.exec(x), y)
        return total_cost / len(test_set)

    def gradient_descent(self, batch: List[Tuple[np.ndarray, np.ndarray]], learning_rate: float,
                         regularization: float, momentum_coefficient: float) -> None:
        """
        Applies gradient descent to the weights and biases in the network.
        :param batch: Given training data.
        :param learning_rate: The learning rate for stochastic gradient descent.
        :param regularization: Regularization parameter.
        :param momentum_coefficient: Momentum co-efficient for the momentum technique.
        """
        # Initialize the gradients.
        gradient_weights = [np.zeros((a, b)) for a, b in zip(self.architecture[1:], self.architecture[:-1])]
        gradient_biases = [np.zeros((a, 1)) for a in self.architecture[1:]]

        # Calculate the gradient of the cost function for each example in the batch.
        for x, y in batch:
            change_weights, change_biases = self.backpropagation(x=x, y=y)
            gradient_weights = [gw + cw for gw, cw in zip(gradient_weights, change_weights)]
            gradient_biases = [gb + cb for gb, cb in zip(gradient_biases, change_biases)]

        # Update weights and biases according to the obtained gradients.
        self.vws = [momentum_coefficient * vw - learning_rate * gw for vw, gw in zip(self.vws, gradient_weights)]
        self.weights = [w - learning_rate * regularization * self.regularization_func(w) + vw
                        for w, vw in zip(self.weights, self.vws)]
        self.vbs = [momentum_coefficient * vb - learning_rate * gb for vb, gb in zip(self.vbs, gradient_biases)]
        self.biases = [b + vb for b, vb in zip(self.biases, self.vbs)]

    def backpropagation(self, x: List[float], y: List[float]) -> (List[float], List[List[float]]):
        """
        Calculates the gradient of the cost function with respect to the network's weights biases for the given training
        example using the four fundamental equations behind backpropagation as described in
        http://neuralnetworksanddeeplearning.com/chap2.html.
        :param x: Input.
        :param y: Correct output.
        :return: The gradient of the cost function in two lists, one for biases and one for weights.
        """
        # Initialize the gradient.
        change_weights = [np.empty((a, b)) for a, b in zip(self.architecture[1:], self.architecture[:-1])]
        change_biases = [np.empty((a, 1)) for a in self.architecture[1:]]

        # Calculate the z and activations for all layers.
        self.exec(x)

        # Compute the gradient for the error in each layer.
        change = None  # Save previous value.
        for l in range(len(self.architecture) - 1, 0, -1):
            if l == len(self.architecture) - 1: # Output layer.
                # Equation 1.
                change = np.multiply(self.cost_func_prime(self.activations[l], y), self.ol_act_func_prime(self.zs[l]))
            else: # Hidden layer
                # Equation 2.
                change = np.multiply(np.matmul(self.weights[l].transpose(), change), self.hl_act_func_prime(self.zs[l]))

            # Equation 3.
            change_biases[l-1] = change
            # Equation 4.
            change_weights[l-1] = np.matmul(change, self.activations[l-1].transpose())

        return change_weights, change_biases
