from neural_net import *


if __name__ == "__main__":

    # Create datasets
    x = [np.random.uniform(-2*np.pi, 2*np.pi, (1,1)) for z in range(1000)]
    y = [np.sin(a) for a in x]
    training_data = [(a, b) for a, b in zip(x, y)]
    x = [np.random.uniform(-2 * np.pi, 2 * np.pi, (1, 1)) for z in range(100)]
    y = [np.sin(a) for a in x]
    test_data = [(a, b) for a, b in zip(x, y)]

    # Create and train the neural net
    nn = NeuralNet([1, 40, 12, 1], hl_act_func=ActivationFunction.TANH, ol_act_func=ActivationFunction.TANH, cost_func=CostFunction.CROSS_ENTROPY, regularization_technique=RegularizationTechnique.L2)
    nn.train(training_data, epochs=500, mini_batch_size=10, learning_rate=0.1, momentum_coefficient=0.9, regularization=0.1, print_progress=True)
    nn.save("networks/sinus-40-12.nn")

    # Test the network
    for e in [0.5, 1, 2, 3, 6, 12, np.inf]:
        print("pi / {0}, True value: {1}, Estimate: {2}".format(e, np.sin(np.pi / e), nn.exec(np.array([[np.pi / e]]))[0][0]))
    print("\nAverage cost:", nn.test(test_data))