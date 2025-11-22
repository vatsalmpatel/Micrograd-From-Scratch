# Micrograd

A tiny scalar-valued autograd engine with a small PyTorch-like neural network library on top.

Micrograd implements **reverse-mode automatic differentiation (backpropagation)** over a dynamically built computational graph. It supports building and training neural networks with a simple, intuitive API similar to PyTorch but operating only on scalar values.

## Features

- ✅ **Automatic Differentiation**: Automatically compute gradients for any scalar-valued function
- ✅ **Dynamic Computation Graph**: Build graphs on-the-fly as operations are performed
- ✅ **Neural Network Building Blocks**: Neurons, layers, and multi-layer perceptrons (MLPs)
- ✅ **PyTorch-like API**: Familiar interface for anyone who has used PyTorch
- ✅ **Educational**: Clean, minimal implementation perfect for learning how autograd works
- ✅ **No Dependencies**: Pure Python implementation (only requires `random` from stdlib)

## Installation

### From Source

1. Clone this repository:
```bash
git clone https://github.com/vatsalmpatel/Micrograd-From-Scratch.git
cd micrograd
```

2. Install in development mode:
```bash
pip install -e .
```

### Requirements

- Python 3.6 or higher
- No external dependencies required for core functionality
- Optional: `torch` for testing/comparison (only in test files)

## Quick Start

### Basic Autograd Example

```python
from grad_code.engine import Value

# Create scalar values
x = Value(2.0)
y = Value(3.0)

# Perform operations
z = x * y + x**2
# z = 2*3 + 2^2 = 6 + 4 = 10

# Compute gradients
z.backward()

print(f"z = {z.data}")      # z = 10.0
print(f"dz/dx = {x.grad}")  # dz/dx = y + 2*x = 3 + 4 = 7.0
print(f"dz/dy = {y.grad}")  # dz/dy = x = 2.0
```

### Neural Network Example

```python
from grad_code.nn import MLP
from grad_code.engine import Value

# Create a multi-layer perceptron
# Architecture: 3 inputs -> 4 hidden -> 4 hidden -> 1 output
model = MLP(3, [4, 4, 1])

# Sample input
x = [Value(1.0), Value(2.0), Value(-1.0)]

# Forward pass
output = model(x)
print(f"Output: {output}")

# Backward pass
output.backward()

# Simple gradient descent update
learning_rate = 0.01
for p in model.parameters():
    p.data -= learning_rate * p.grad

# Reset gradients for next iteration
model.zero_grad()
```

### Training a Neural Network

Here's a complete example training a simple network on a toy dataset:

```python
from grad_code.nn import MLP
from grad_code.engine import Value

# Create dataset (XOR-like problem)
X = [
    [Value(0.0), Value(0.0)],
    [Value(0.0), Value(1.0)],
    [Value(1.0), Value(0.0)],
    [Value(1.0), Value(1.0)],
]
y = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]

# Create model
model = MLP(2, [4, 1])

# Training loop
learning_rate = 0.1
for epoch in range(100):
    # Forward pass
    predictions = [model(x) for x in X]
    
    # Compute loss (MSE)
    loss = sum((pred - target)**2 for pred, target in zip(predictions, y))
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Update parameters
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

# Test the trained model
print("\nPredictions:")
for i, x in enumerate(X):
    pred = model(x)
    print(f"Input: {[v.data for v in x]}, Target: {y[i].data}, Prediction: {pred.data}")
```

## API Reference

### `Value` Class (grad_code.engine)

The core autograd engine that wraps scalar values and tracks operations.

#### Constructor
```python
Value(data, _children=(), _op='')
```
- `data`: The scalar value to store
- `_children`: Tuple of parent Value objects (for internal use)
- `_op`: String describing the operation (for debugging)

#### Attributes
- `data`: The scalar value
- `grad`: The gradient of this value (computed by `.backward()`)

#### Methods
- `backward()`: Compute gradients for all values in the computation graph

#### Supported Operations
- **Arithmetic**: `+`, `-`, `*`, `/` (with Value or scalar)
- **Power**: `**` (supports int/float exponents)
- **Activation**: `.relu()` - ReLU activation function
- **Negation**: `-value`

### Neural Network Classes (grad_code.nn)

#### `Module`
Base class for all neural network components.

**Methods:**
- `parameters()`: Returns list of all trainable parameters
- `zero_grad()`: Resets all parameter gradients to zero

#### `Neuron`
A single neuron with weights, bias, and optional ReLU activation.

```python
Neuron(nin, nonlin=True)
```
- `nin`: Number of inputs
- `nonlin`: Whether to apply ReLU (True) or be linear (False)

#### `Layer`
A layer of neurons.

```python
Layer(nin, nout, **kwargs)
```
- `nin`: Number of inputs to each neuron
- `nout`: Number of neurons in the layer
- `**kwargs`: Additional arguments (e.g., `nonlin=False`)

#### `MLP`
Multi-layer perceptron (feedforward neural network).

```python
MLP(nin, nouts)
```
- `nin`: Number of input features
- `nouts`: List of layer sizes, e.g., `[16, 16, 1]` for 2 hidden layers and 1 output

**Example:**
```python
# 3 inputs -> 16 hidden -> 1 output
model = MLP(3, [16, 1])
```

## How It Works

### Automatic Differentiation

Micrograd uses **reverse-mode automatic differentiation** (also known as backpropagation):

1. **Forward Pass**: As you perform operations, a computation graph is built where:
   - Each `Value` object is a node
   - Edges represent dependencies between operations

2. **Backward Pass**: When you call `.backward()`:
   - The graph is traversed in topological order (from output to inputs)
   - Gradients are computed using the chain rule
   - Each operation knows how to propagate gradients to its inputs

### Example Computation Graph

```python
x = Value(2.0)
y = Value(3.0)
z = x * y + x**2
```

Creates the graph:
```
    x (2.0)  ────┬────> * ────┐
                 │            │
    y (3.0)  ────┘            ├───> + ────> z (10.0)
                              │
    x (2.0)  ──> **2 ─────────┘
```

When `z.backward()` is called:
- `z.grad = 1` (by definition, dz/dz = 1)
- Gradients flow backward through the graph
- Chain rule is applied at each operation
- Final gradients: `x.grad = 7.0`, `y.grad = 2.0`

### Supported Operations and Their Gradients

| Operation | Forward | Backward (gradient) |
|-----------|---------|-------------------|
| Addition | `z = x + y` | `dx = dz`, `dy = dz` |
| Multiplication | `z = x * y` | `dx = y * dz`, `dy = x * dz` |
| Power | `z = x ** n` | `dx = n * x^(n-1) * dz` |
| ReLU | `z = max(0, x)` | `dx = (x > 0) * dz` |
| Division | `z = x / y` | Implemented as `x * y^(-1)` |
| Subtraction | `z = x - y` | Implemented as `x + (-y)` |

## Testing

Run the test suite to verify the implementation:

```bash
# Run tests with pytest
pytest test/test_engine.py

# Or run directly with Python (after setting PYTHONPATH)
set PYTHONPATH=d:\ML\micrograd
python test/test_engine.py
```

The tests compare micrograd's gradients against PyTorch to ensure correctness.

## Project Structure

```
micrograd/
├── grad_code/           # Main package
│   ├── __init__.py     # Package initialization
│   ├── engine.py       # Autograd engine (Value class)
│   └── nn.py           # Neural network components
├── test/               # Test suite
│   └── test_engine.py  # Tests comparing against PyTorch
├── setup.py            # Package installation configuration
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Educational Value

This project is ideal for:

- **Learning how autograd works**: See a complete, minimal implementation of automatic differentiation
- **Understanding backpropagation**: Every operation clearly shows how gradients flow backward
- **Building intuition for PyTorch**: The API mirrors PyTorch's design philosophy
- **Teaching neural networks**: Small enough to understand completely, powerful enough to be useful

## Limitations

- **Scalar-only**: Only operates on scalar values, not tensors/matrices
- **No GPU support**: Pure Python, runs on CPU only  
- **Basic operations**: Limited to fundamental operations (add, mul, pow, relu)
- **Not optimized**: Designed for clarity and education, not performance
- **No advanced features**: No batch processing, convolutions, attention, etc.

For production use, consider PyTorch, TensorFlow, or JAX.

## Credits

This implementation is inspired by and based on [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd), created as an educational tool to understand the fundamentals of automatic differentiation and neural networks.

## License

MIT License

Copyright (c) 2025 Vatsal Patel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Resources

- [Original micrograd by Andrej Karpathy](https://github.com/karpathy/micrograd)
- [Video explanation](https://www.youtube.com/watch?v=VMj-3S1tku0) - Andrej's YouTube tutorial
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) - 3Blue1Brown

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add more examples
- Improve documentation

Please open an issue or submit a pull request on GitHub.
