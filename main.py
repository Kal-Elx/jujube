from neural_net import *

def main():
    nn = NeuralNet([2, 4, 2], ActivationFunction.SIGMOID)
    print(nn.exec([5, 8]))

if __name__ == "__main__":
    main()
