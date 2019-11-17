from enum import Enum
import numpy as np
from typing import List, Tuple, Callable
from random import shuffle
import pickle
import warnings
import time


class ActivationFunction(Enum):
    SIGMOID = 1
    TANH = 2
    RELU = 3
    LINEAR = 4


def get_activation_func(activation_func: ActivationFunction) -> Tuple[Callable, Callable]:
    """
    Returns the corresponding activation function and its derivative for the given enum.
    :param activation_func: Desired activation function.
    :return: Tuple of desired activation function and its derivative.
    """
    if activation_func == ActivationFunction.SIGMOID:
        return sigmoid, sigmoid_prime
    elif activation_func == ActivationFunction.TANH:
        return np.tanh, tanh_prime
    elif activation_func == ActivationFunction.RELU:
        return relu, relu_prime
    elif activation_func == ActivationFunction.LINEAR:
        return linear, linear_prime


class CostFunction(Enum):
    QUADRATIC_COST = 1


def get_cost_func(cost_func: CostFunction) -> Tuple[Callable, Callable]:
    """
    Returns the corresponding cost function and its derivative for the given enum.
    :param cost_func: Desired cost function.
    :return: Tuple of desired activation function and its derivative.
    """
    if cost_func == CostFunction.QUADRATIC_COST:
        return quadratic_cost, quadratic_cost_prime


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x: float) -> float:
    return sigmoid(x)*(1-sigmoid(x))


def tanh_prime(x: float) -> float:
    # TODO: Implement
    pass


def relu(x: float) -> float:
    # TODO: Implement
    pass


def relu_prime(x: float) -> float:
    # TODO: Implement
    pass


def linear(x: float) -> float:
    return x


def linear_prime(x: float) -> float:
    return 1


def quadratic_cost(a: float, y: float) -> float:
    # TODO: Implement
    pass


def quadratic_cost_prime(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return a - y
