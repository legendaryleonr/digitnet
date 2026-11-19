"""Optimizers: plain SGD with momentum, and Adam."""
import numpy as np


class SGD:
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0):
        self.params = params
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.v = [np.zeros_like(p) for p, _ in params]

    def step(self):
        for i, (p, g) in enumerate(self.params):
            g = g + self.wd * p
            self.v[i] = self.momentum * self.v[i] - self.lr * g
            p += self.v[i]


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, (p, g) in enumerate(self.params):
            g = g + self.wd * p
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            # bias correction: early steps would otherwise be too small
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
