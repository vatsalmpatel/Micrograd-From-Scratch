"""
Micrograd: A Tiny Scalar-Valued Autograd Engine

A minimalist automatic differentiation library with a PyTorch-like API.
Micrograd implements reverse-mode automatic differentiation (backpropagation)
over a dynamically built computational graph.

Core modules:
    - engine: The autograd engine with the Value class
    - nn: Neural network building blocks (Neuron, Layer, MLP)

Example:
    >>> from grad_code.engine import Value
    >>> from grad_code.nn import MLP
    >>> 
    >>> # Simple autograd example
    >>> x = Value(2.0)
    >>> y = x ** 2 + 2 * x + 1
    >>> y.backward()
    >>> print(x.grad)  # dy/dx = 2*x + 2 = 6.0
    >>> 
    >>> # Neural network example
    >>> model = MLP(3, [4, 4, 1])
    >>> x = [Value(1.0), Value(2.0), Value(-1.0)]
    >>> output = model(x)
"""

from grad_code.engine import Value
from grad_code.nn import Neuron, Layer, MLP, Module

__all__ = ['Value', 'Neuron', 'Layer', 'MLP', 'Module']
__version__ = '0.0.0'
