from neural_net import *
import mnist_loader

def mnist(hidden_layer_architecture: List[int], epochs: int, mini_batch_size: int, learning_rate: float,
          training_examples: int = 50000, test_examples: int = 10000) -> None:

    # Prepare the data set.
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)[:training_examples]
    test_data = list(test_data)[:test_examples]

    # Initialize the neural net.
    nn = NeuralNet([784] + hidden_layer_architecture + [10], print_progress=True)

    # Train the neural net.
    nn.train(training_set=training_data, epochs=epochs, mini_batch_size=mini_batch_size, learning_rate=learning_rate)

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
    mnist(hidden_layer_architecture=[32, 32], epochs=30, mini_batch_size=10, learning_rate=3.0)
