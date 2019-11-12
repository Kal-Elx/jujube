from enum import Enum
import numpy as np
from typing import List, Tuple, Callable
from random import shuffle


class ActivationFunction(Enum):
    SIGMOID = 1
    TANH = 2
    RELU = 3
    LINEAR = 4


def get_activation_func(activation_func: ActivationFunction) -> Callable:
    if activation_func == ActivationFunction.SIGMOID:
        return sigmoid
    elif activation_func == ActivationFunction.TANH:
        return np.tanh
    elif activation_func == ActivationFunction.RELU:
        return relu
    elif activation_func == ActivationFunction.LINEAR:
        return linear


def get_activation_func_prime(activation_func: ActivationFunction) -> Callable:
    if activation_func == ActivationFunction.SIGMOID:
        return sigmoid_prime
    elif activation_func == ActivationFunction.TANH:
        return tanh_prime
    elif activation_func == ActivationFunction.RELU:
        return relu_prime
    elif activation_func == ActivationFunction.LINEAR:
        return linear_prime


class CostFunction(Enum):
    QUADRATIC_COST = 1


def get_cost_func(cost_func: CostFunction) -> Callable:
    if cost_func == CostFunction.QUADRATIC_COST:
        return quadratic_cost


def get_cost_func_prime(cost_func: CostFunction) -> Callable:
    if cost_func == CostFunction.QUADRATIC_COST:
        return quadratic_cost_prime


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x: float) -> float:
    return sigmoid(x)*(1-sigmoid(x))


def tanh_prime(x: float) -> float:
    # TODO: Implement
    return 0.0


def relu(x: float) -> float:
    if x > 1:
        return 1
    elif x < 0:
        return 0
    else:
        return x


def relu_prime(x: float) -> float:
    if x > 1 or x < 0:
        return 0
    else:
        return 1


def linear(x: float) -> float:
    return x


def linear_prime(x: float) -> float:
    return 1


def quadratic_cost(a: List[float], y: List[float]) -> float:
    assert len(y) == len(a)
    return sum(x ** 2 for x in [y[i] - a[i] for i in range(len(a))])


def quadratic_cost_prime(a: List[float], y: List[float]) -> List[float]:
    assert len(y) == len(a)
    return [a-y for a, y in zip(a, y)]


def transpose(m: List[List[float]]) -> List[List[float]]:
    return [list(e) for e in zip(*m)]