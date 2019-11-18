from neural_net import *
import mnist_loader


def new(hidden_layer_architecture: List[int], epochs: int, mini_batch_size: int, learning_rate: float,
        training_data: List[Tuple[np.ndarray, np.ndarray]], regularization_param: float = 0.0) -> NeuralNet:
    # Initialize the neural net.
    nn = NeuralNet([784] + hidden_layer_architecture + [10], cost_func=CostFunction.CROSS_ENTROPY,
                   regularization_technique=RegularizationTechnique.L2)

    # Train the neural net.
    nn.train(training_set=training_data, epochs=epochs, mini_batch_size=mini_batch_size, learning_rate=learning_rate,
             regularization_param=regularization_param, print_progress=True)

    # Save the neural net to file.
    nn.save('examples/mnist-{0}.nn'.format(str(hidden_layer_architecture)).replace('[', '').replace(']', '')
            .replace(',', '').replace(' ', '-'))

    return nn


def test(nn: NeuralNet, test_data: List[Tuple[np.ndarray, np.ndarray]]):
    # Test the neural net.
    correct = 0.0
    print()
    for x, y in test_data:
        output = nn.exec(x)
        guess = 0
        for i in range(10):
            if output[i][0] > output[guess][0]:
                guess = i
        if guess != y:
            print("Correct: {0}, Guess: {1}, Output: {2}".format(y, guess, str(output.flatten()).replace('\n', ' ')))
        correct += (guess == y)

    print("\nAccuracy: {0}%".format((correct / len(test_data) * 100)))


if __name__ == "__main__":

    # Prepare the data set.
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    validation_data = list(validation_data)
    test_data = list(test_data)

    # Create new neural net or load an existing one.
    nn = new(hidden_layer_architecture=[100], epochs=5, mini_batch_size=10, learning_rate=0.1,
             regularization_param=5.0, training_data=training_data)
    #nn = NeuralNet.load('examples/mnist-30.nn')

    # Test the neural net.
    test(nn=nn, test_data=test_data)
    #print("Average cost: {0}".format(nn.test(training_data[:1000])))
