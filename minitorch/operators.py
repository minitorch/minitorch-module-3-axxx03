"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

EPS = 1e-5

# ## Task 0.1 Basic hypothesis tests.
def add(x: float, y: float):
    res = x + y
    return res

def eq(x: float, y: float):
    if math.fabs(x - y) < EPS:
        return 1.0
    else:
        return 0.0

def id(x: float):
    # id - Returns the input unchanged
    return x

def inv(x: float):
    # inv - Calculates the reciprocal
    return 1.0 / x

def inv_back(x: float, y: float):
    # inv_back - Computes the derivative of reciprocal times a second arg
    return -y / x**2

def log(x: float):
    return math.log(x)

def exp(x: float):
    return math.exp(x)

def exp_back(x: float, y: float):
    # exp_back - Computes the derivative of exp times a second arg
    return y * exp(x)

def log_back(x: float, y: float):
    # log_back - Computes the derivative of log times a second arg
    return y / x

def lt(x: float, y: float):
    # lt - Checks if one number is less than another
    if eq(x, y):
        return 0.0
    return 1.0 if x < y else 0.0

def leq(x: float, y: float):
    # x <= y
    return lt(x, y) or eq(x, y)

def max(x: float, y: float):
    return x if x > y else y

def mul(x: float, y: float):
    return x * y

def neg(x: float):
    return -1.0 * x

def prod(list1: List):
    res = 1.0
    for x in list1:
        res *= x
    return res

def relu(x: float):
    return max(0.0, x)

def relu_back(x: float, y: float):
    # relu_back - Computes the derivative of ReLU times a second arg
    return 1.0 * y if x > 0.0 else 0.0


def sigmoid(x: float):
    # sigmoid - Calculates the sigmoid function
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_back(x: float, y: float):
    # sigmoid_back - Computes the derivative of sigmoid times a second arg
    return sigmoid(x) * (1 - sigmoid(x)) * y

def is_close(x: float, y: float):
    return math.fabs(x - y) < EPS

# ## Task 0.3  - Higher-order functions
# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce

def map(fn: Callable, list1: List):
    return [fn(x) for x in list1]

def zipWith(fn: Callable, list1: List, list2: List):
    return [fn(x, y) for x, y in zip(list1, list2)]

def reduce(fn: Callable, list1: List):
    if len(list1) == 0:
        return 0.0
    
    result = list1[0]
    for i in range(1, len(list1)):
        result = fn(result, list1[i])
    return result

# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def negList(list1: List):
    return map(neg, list1)

def addLists(list1: List, list2: List):
    return zipWith(add, list1, list2)

def sum(list1: List):
    return reduce(add, list1)

def prod(list1: List):
    return reduce(mul, list1)