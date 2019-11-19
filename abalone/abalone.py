import csv
from neural_net import *

if __name__ == "__main__":

    # Create data sets.
    with open('abalone.data') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        for row in csv_reader:
            sex = 2.0 if row[0] == 'M' else 1.0 if row[0] == 'F' else 0.0
            data.append((np.array([[sex]] + [[float(x)] for x in row[1:-1]]), np.array([[float(row[-1])]])))
    training_data = data[:4000]
    test_data = data[4000:]

    # Create and train the neural net.
    nn = NeuralNet([8, 40, 12, 1], ol_act_func=ActivationFunction.LINEAR, regularization_technique=RegularizationTechnique.L2)
    nn.train(training_data, epochs=100, mini_batch_size=10, learning_rate=0.1, regularization=0.5, momentum_coefficient=0.5, print_progress=True)
    nn.save("networks/abalone-40-12.nn")

    # Test the neural net.
    print("Average cost:", nn.test(test_data))