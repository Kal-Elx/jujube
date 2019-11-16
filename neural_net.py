from utils import *


class NeuralNet:

    def __init__(self, architecture: List[int], hl_act_func: ActivationFunction = ActivationFunction.SIGMOID,
                 ol_act_func: ActivationFunction = ActivationFunction.SIGMOID,
                 cost_func: CostFunction = CostFunction.QUADRATIC_COST, print_progress: bool = False):
        """
        :param architecture: Number of neurons in each net. [input layer, hidden layer, ..., output layer].
        :param hl_act_func: Activation function for the hidden layers.
        :param ol_act_func: Activation function for the output layer.
        """
        # Test given architecture.
        assert len(architecture) >= 2, "The network needs two have at least two layers."
        assert not any(e == 0 for e in architecture), "The network can not have layers with zero neurons."

        # Issue warning for using linear activation function.
        if hl_act_func == ActivationFunction.LINEAR:
            warnings.warn("Linear function is not recommended as activation function in hidden layers.")

        # Save information about the network´s architecture.
        self.architecture: List[int] = architecture

        # Initialize network.
        self.weights = [np.random.randn(a, b) for a, b in zip(self.architecture[1:], self.architecture[:-1])]
        self.biases = [np.random.randn(a, 1) for a in self.architecture[1:]]
        self.zs = [np.empty((a, 1)) for a in self.architecture]
        self.activations = [np.empty((a, 1)) for a in self.architecture]

        # Initialize functions.
        self.hl_act_func, self.hl_act_func_prime = get_activation_func(activation_func=hl_act_func)
        self.ol_act_func, self.ol_act_func_prime = get_activation_func(activation_func=ol_act_func)
        self.cost_func, self.cost_func_prime = get_cost_func(cost_func=cost_func)

        self.print_progress = print_progress

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

        # Calculate the activations int the network in a feedforward manner.
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
                self.gradient_descent(batch=mini_batch, learning_rate=learning_rate / mini_batch_size)

                if self.print_progress and j % 100 == 0:
                    print("Epoch: {0}/{1}, Mini batch: {2}/{3}".format(i + 1, epochs, j, len(mini_batches)))

        if self.print_progress:
            print("\nTraining time: {0} min {1} sec".format(round((time.time()-start_time) // 60), round(time.time()-start_time) % 60))

    def gradient_descent(self, batch: List[Tuple[np.ndarray, np.ndarray]], learning_rate: float) -> None:
        """
        Applies gradient descent to the weights and biases in the network.
        :param batch: Given training data.
        :param learning_rate: The learning rate for stochastic gradient descent.
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
        self.weights = [w - learning_rate * gw for w, gw in zip(self.weights, gradient_weights)]
        self.biases = [b - learning_rate * gb for b, gb in zip(self.biases, gradient_biases)]

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
