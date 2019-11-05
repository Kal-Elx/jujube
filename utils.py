from enum import Enum
import numpy as np
from math import exp, tanh
from typing import List, Callable

class ActivationFunction(Enum):
    SIGMOID = 1
    TANH = 2
    RELU = 3
    LINEAR = 4


def sigmoid(x: float) -> float:
    return 1/(1+exp(-x))


def relu(x: float) -> float:
    if x > 1:
        return 1
    elif x < 0:
        return 0
    else:
        return x


def linear(x: float) -> float:
    return x