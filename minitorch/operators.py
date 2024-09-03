"""Collection of the core mathematical operators used throughout the code base."""

from typing import Callable, Iterable
import math

# ## Task 0.1

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


def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negates a number."""
    return -1.0 * x


def lt(x: float, y: float) -> bool:
    """Returns whether x is less than y."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Returns whether x is equal to y."""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the maximum of x and y."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Returns whether x is within 1e-2 of y."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Applies the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU function."""
    return max(0.0, x)


def log(x: float) -> float:
    """Applies the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Applies the exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Returns the reciprocal of x."""
    return 1.0 / x


def log_back(x: float, grad: float) -> float:
    """Backward log."""
    return 1.0 / x * grad


def inv_back(x: float, grad: float) -> float:
    """Backward inv."""
    return -1.0 / (x * x) * grad


def relu_back(x: float, grad: float) -> float:
    """Backward relu."""
    return grad if x > 0.0 else 0.0


def sigmoid_back(x: float, grad: float) -> float:
    """Backward sigmoid."""

    s = sigmoid(x)
    return s * (1 - s) * grad


def exp_back(x: float, grad: float) -> float:
    """Backward exp."""

    return grad * math.exp(x)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element of an iterable"""

    def _inner(xs: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in xs]

    return _inner


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combines elements of two iterables with given function"""

    def _inner(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(xs, ys)]

    return _inner


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduces an iterable to a single value using a given function"""

    def _inner(xs: Iterable[float]) -> float:
        total = start
        for x in xs:
            total = fn(x, total)

        return total

    return _inner


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list."""
    return map(neg)(xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists."""
    return zipWith(add)(xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum all elements in a list."""
    return reduce(add, 0.0)(xs)


def prod(xs: Iterable[float]) -> float:
    """Calculate the product of all elements in a list."""
    return reduce(mul, 1.0)(xs)
