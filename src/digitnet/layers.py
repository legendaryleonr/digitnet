"""Neural network layers implemented with NumPy only.

Every layer has two methods:
  forward(x)  -> computes the output and remembers what it needs for backprop
  backward(g) -> takes dLoss/dOutput and returns dLoss/dInput,
                 storing gradients for its own parameters along the way
"""
import numpy as np


class Layer:
    def forward(self, x, training=True):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError

    def params(self):
        """Return list of (parameter, gradient) pairs. Empty for layers with no weights."""
        return []


class Dense(Layer):
    """Fully connected layer: y = x @ W + b"""

    def __init__(self, n_in, n_out, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        # He initialization: keeps activations from shrinking/exploding with ReLU
        self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x, training=True):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.dW[...] = self.x.T @ grad_out
        self.db[...] = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU(Layer):
    def forward(self, x, training=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask


class Dropout(Layer):
    """Randomly zeroes activations during training to reduce overfitting."""

    def __init__(self, p=0.2, rng=None):
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng(1)

    def forward(self, x, training=True):
        if not training or self.p == 0:
            self.mask = None
            return x
        # "inverted" dropout: scale at train time so inference needs no change
        self.mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad_out):
        return grad_out if self.mask is None else grad_out * self.mask
