"""
Micrograd: Neural Network Components

This module provides PyTorch-like neural network building blocks built on top
of the autograd engine. It includes neurons, layers, and multi-layer perceptrons
(MLPs) that can be composed to build neural networks.

All classes inherit from Module, which provides common functionality like
parameter management and gradient zeroing.
"""

import random
from grad_code.engine import Value


class Module:
    """
    Base class for all neural network modules.
    
    This is similar to torch.nn.Module and provides the basic interface
    that all neural network components should implement. Subclasses should
    override parameters() to return their trainable parameters.
    """

    def zero_grad(self):
        """
        Reset gradients of all parameters to zero.
        
        This should be called before each backward pass during training to
        prevent gradient accumulation across multiple backward passes.
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """
        Return a list of all trainable parameters in this module.
        
        Returns:
            list: Empty list (subclasses should override this)
        """
        return []


class Neuron(Module):
    """
    A single neuron with optional ReLU activation.
    
    A neuron performs a weighted sum of its inputs plus a bias, optionally
    followed by a ReLU non-linearity. This is the atomic building block of
    neural networks.
    
    Attributes:
        w (list): List of Value objects representing weights
        b (Value): Bias term
        nonlin (bool): Whether to apply ReLU activation
        
    Example:
        >>> neuron = Neuron(3)  # 3 inputs
        >>> x = [Value(1.0), Value(2.0), Value(3.0)]
        >>> output = neuron(x)
    """

    def __init__(self, nin, nonlin=True):
        """
        Initialize a neuron.
        
        Args:
            nin (int): Number of inputs to this neuron
            nonlin (bool): If True, apply ReLU activation; if False, linear neuron
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        """
        Perform a forward pass through the neuron.
        
        Computes: activation = sum(w_i * x_i) + b
        Then applies ReLU if nonlin=True.
        
        Args:
            x (list): List of Value objects or scalars as inputs
            
        Returns:
            Value: The neuron's output
        """
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        """
        Return all trainable parameters (weights and bias).
        
        Returns:
            list: All weight Values plus the bias Value
        """
        return self.w + [self.b]

    def __repr__(self):
        """Return a string representation of this neuron."""
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """
    A layer of neurons.
    
    A layer consists of multiple neurons that all take the same input.
    Each neuron in the layer produces one output, so a layer with nout neurons
    produces nout outputs.
    
    Attributes:
        neurons (list): List of Neuron objects in this layer
        
    Example:
        >>> layer = Layer(3, 4)  # 3 inputs, 4 outputs
        >>> x = [Value(1.0), Value(2.0), Value(3.0)]
        >>> outputs = layer(x)  # Returns list of 4 Value objects
    """

    def __init__(self, nin, nout, **kwargs):
        """
        Initialize a layer of neurons.
        
        Args:
            nin (int): Number of inputs to each neuron
            nout (int): Number of neurons in this layer (= number of outputs)
            **kwargs: Additional arguments passed to each Neuron (e.g., nonlin=False)
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """
        Perform a forward pass through all neurons in the layer.
        
        Args:
            x (list): Input values
            
        Returns:
            Value or list: Single Value if layer has 1 neuron, otherwise list of Values
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """
        Return all trainable parameters in this layer.
        
        Returns:
            list: Flattened list of all parameters from all neurons
        """
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        """Return a string representation of this layer."""
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """
    Multi-Layer Perceptron (MLP) - a feedforward neural network.
    
    An MLP consists of multiple layers stacked sequentially. The output of
    each layer becomes the input to the next layer. The final layer uses
    linear activation (no ReLU) by default, which is standard for regression
    or before applying a different loss function for classification.
    
    Attributes:
        layers (list): List of Layer objects
        
    Example:
        >>> # Build a network: 3 inputs -> 4 hidden -> 4 hidden -> 1 output
        >>> model = MLP(3, [4, 4, 1])
        >>> x = [Value(1.0), Value(2.0), Value(3.0)]
        >>> output = model(x)
        >>> 
        >>> # Training step
        >>> output.backward()
        >>> for p in model.parameters():
        ...     p.data -= 0.01 * p.grad  # Simple gradient descent
        >>> model.zero_grad()
    """

    def __init__(self, nin, nouts):
        """
        Initialize a multi-layer perceptron.
        
        Args:
            nin (int): Number of input features
            nouts (list): List of output sizes for each layer.
                         For example, [16, 16, 1] creates a network with
                         two hidden layers of 16 neurons and an output layer of 1 neuron.
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i != len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Perform a forward pass through all layers.
        
        Args:
            x (list): Input values
            
        Returns:
            Value or list: Network output (depends on output layer size)
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Return all trainable parameters in the network.
        
        Returns:
            list: Flattened list of all parameters from all layers
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """Return a string representation of this MLP."""
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"