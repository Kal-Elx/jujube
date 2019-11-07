from enum import Enum
import numpy as np
from math import exp, tanh
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


class CostFunction(Enum):
    QUADRATIC_COST = 1


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def relu(x: float) -> float:
    if x > 1:
        return 1
    elif x < 0:
        return 0
    else:
        return x


def linear(x: float) -> float:
    return x


def quadratic_cost(y: List[float], a: List[float]) -> float:
    assert len(y) == len(a)
    return sum(x ** 2 for x in [y[i] - a[i] for i in range(len(a))])
