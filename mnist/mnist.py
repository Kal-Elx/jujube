import csv
import gzip
from neural_net import *


def new(hidden_layer_architecture: List[int], epochs: int, mini_batch_size: int, learning_rate: float,
        training_data: List[Tuple[np.ndarray, np.ndarray]], regularization_param: float = 0.0,
        momentum_coefficient: float = 0.0) -> NeuralNet:
    # Initialize the neural net.
    nn = NeuralNet([784] + hidden_layer_architecture + [10], cost_func=CostFunction.CROSS_ENTROPY,
                   regularization_technique=RegularizationTechnique.L2)

    # Train the neural net.
    nn.train(training_set=training_data, epochs=epochs, mini_batch_size=mini_batch_size, learning_rate=learning_rate, print_progress=True)

    # Save the neural net to file.
    nn.save('examples/mnist-{0}.nn'.format(str(hidden_layer_architecture)).replace('[', '').replace(']', '')
            .replace(',', '').replace(' ', '-'))

    return nn


def test(nn: NeuralNet, test_data: List[Tuple[np.ndarray, np.ndarray]]):

    correct = 0.0
    print()
    for x, y in test_data:
        output = nn.exec(x)
        guess = 0
        ans = 0
        for i in range(10):
            if output[i][0] > output[guess][0]:
                guess = i
            if y[i][0] > y[ans][0]:
                ans = i
        if guess != ans:
            print("Correct: {0}, Guess: {1}, Output: {2}".format(ans, guess, str(output.flatten()).replace('\n', ' ')))
        correct += (guess == ans)

    print("\nAccuracy: {0}%".format((correct / len(test_data) * 100)))


def parse_mnist(input_file: str, output_file: str):
    first = True
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        for row in csv_reader:
            if first:
                first = False
                continue

            x = np.array([[rescale(0.0, 255.0, 0.0, 1.0, float(x))] for x in row[1:]])
            y = np.full((10, 1), 0.0)
            y[int(row[0])] = 1.0
            data.append((x, y))

    filehandler = gzip.open(output_file, 'wb')
    pickle.dump(data, filehandler)
    filehandler.close()


if __name__ == "__main__":

    # Prepare the data set.
    parse_mnist('mnist_train.csv', 'training_data.pkl.gzip') # Files to large for git
    filehandler = gzip.open('training_data.pkl.gzip', 'rb')
    training_data = pickle.load(filehandler)
    filehandler.close()

    parse_mnist('mnist_test.csv', 'test_data.pkl.gzip') # Files to large for git
    filehandler = gzip.open('test_data.pkl.gzip', 'rb')
    test_data = pickle.load(filehandler)
    filehandler.close()

    # Create new neural net or load an existing one.
    nn = new(hidden_layer_architecture=[30], epochs=30, mini_batch_size=10, learning_rate=3.0, training_data=training_data[:1000])
    # nn = NeuralNet.load('examples/mnist-100.nn')

    # Test the neural net.
    test(nn=nn, test_data=test_data[:100])
