"""Softmax + cross-entropy loss, combined for numerical stability."""
import numpy as np


def softmax(logits):
    # subtract the row max so exp() never overflows
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        """logits: (N, C) raw scores, y: (N,) integer labels. Returns mean loss."""
        self.probs = softmax(logits)
        self.y = y
        n = logits.shape[0]
        return -np.log(self.probs[np.arange(n), y] + 1e-12).mean()

    def backward(self):
        # The gradient of softmax+cross-entropy simplifies to (probs - one_hot) / N
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.y] -= 1.0
        return grad / n
