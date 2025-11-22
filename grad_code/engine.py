"""
Micrograd: A Tiny Autograd Engine

This module implements the core automatic differentiation engine that powers
the neural network library. It provides a scalar-valued autograd system that
can track operations and compute gradients using backpropagation.

The Value class wraps scalar values and tracks their computational history,
enabling automatic differentiation through the backward() method.
"""


class Value:
    """
    Stores a single scalar value and its gradient.
    
    The Value class is the core building block of the autograd engine. It wraps
    a scalar value and tracks all operations performed on it, building a computational
    graph that can be traversed in reverse to compute gradients via backpropagation.
    
    Attributes:
        data (float): The actual scalar value stored in this node
        grad (float): The gradient of this value with respect to some output
        _prev (set): Set of Value objects that are inputs to this node
        _op (str): The operation that produced this node (for debugging/visualization)
        _backward (callable): Function to propagate gradients to parent nodes
        
    Example:
        >>> x = Value(2.0)
        >>> y = Value(3.0)
        >>> z = x * y + x**2
        >>> z.backward()
        >>> print(x.grad)  # dz/dx = y + 2*x = 3.0 + 2*2.0 = 7.0
        7.0
    """

    def __init__(self, data, _children=(), _op=''):
        """
        Initialize a Value node.
        
        Args:
            data (float): The scalar value to store
            _children (tuple): Tuple of Value objects that are inputs to this node
            _op (str): String describing the operation that created this node
        """
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        """
        Add two values (handles both Value + Value and Value + scalar).
        
        Implements the forward pass for addition and defines the backward pass
        for gradient propagation. The gradient flows equally to both operands.
        
        Args:
            other (Value or float): The value to add
            
        Returns:
            Value: A new Value object containing the sum
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Multiply two values (handles both Value * Value and Value * scalar).
        
        Implements the forward pass for multiplication and defines the backward pass.
        The gradient to each operand is the output gradient times the other operand.
        
        Args:
            other (Value or float): The value to multiply by
            
        Returns:
            Value: A new Value object containing the product
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        """
        Raise this value to a power (only supports int/float exponents).
        
        Implements the forward pass for exponentiation and defines the backward pass
        using the power rule: d/dx(x^n) = n * x^(n-1)
        
        Args:
            other (int or float): The exponent
            
        Returns:
            Value: A new Value object containing self raised to the power of other
            
        Raises:
            AssertionError: If other is not an int or float
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """
        Apply the ReLU (Rectified Linear Unit) activation function.
        
        ReLU(x) = max(0, x). The gradient is 1 when x > 0, and 0 otherwise.
        This is a common activation function in neural networks that introduces
        non-linearity while being computationally efficient.
        
        Returns:
            Value: A new Value object with ReLU applied
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """
        Compute gradients for all nodes in the computational graph.
        
        This method performs backpropagation by:
        1. Building a topological ordering of all nodes in the computation graph
        2. Initializing this node's gradient to 1 (since d(self)/d(self) = 1)
        3. Traversing the graph in reverse topological order, calling each node's
           _backward function to propagate gradients to its inputs
           
        This implements reverse-mode automatic differentiation (backpropagation).
        After calling backward(), all Value objects in the computation graph will
        have their .grad attribute set to the derivative of this output with respect
        to that value.
        
        Example:
            >>> x = Value(2.0)
            >>> y = x * 3 + x**2
            >>> y.backward()
            >>> print(x.grad)  # dy/dx = 3 + 2*x = 3 + 2*2 = 7
            7.0
        """

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        """Return the negation of this value."""
        return self * -1

    def __radd__(self, other):  # other + self
        """Handle addition when this Value is on the right side."""
        return self + other

    def __sub__(self, other):  # self - other
        """Subtract another value from this value."""
        return self + (-other)

    def __rsub__(self, other):  # other - self
        """Handle subtraction when this Value is on the right side."""
        return other + (-self)

    def __rmul__(self, other):  # other * self
        """Handle multiplication when this Value is on the right side."""
        return self * other

    def __truediv__(self, other):  # self / other
        """Divide this value by another value."""
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        """Handle division when this Value is on the right side."""
        return other * self**-1

    def __repr__(self):
        """Return a string representation of this Value."""
        return f"Value(data={self.data}, grad={self.grad})"