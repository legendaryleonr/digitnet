"""A small sequential model container plus save/load helpers."""
import json
import numpy as np

from .layers import Dense, ReLU, Dropout


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self):
        return [pg for layer in self.layers for pg in layer.params()]

    def predict(self, x):
        return self.forward(x, training=False).argmax(axis=1)

    def export_json(self, path, meta=None):
        """Save the Dense weights so the browser demo can run the same network."""
        dense = [l for l in self.layers if isinstance(l, Dense)]
        payload = {
            "layers": [{"W": l.W.round(6).tolist(), "b": l.b.round(6).tolist()} for l in dense],
            "meta": meta or {},
        }
        with open(path, "w") as f:
            json.dump(payload, f)


def build_mlp(n_in=64, hidden=(128, 64), n_out=10, dropout=0.2, seed=0):
    rng = np.random.default_rng(seed)
    layers, prev = [], n_in
    for h in hidden:
        layers += [Dense(prev, h, rng), ReLU()]
        if dropout > 0:
            layers.append(Dropout(dropout, np.random.default_rng(seed + 100 + h)))
        prev = h
    layers.append(Dense(prev, n_out, rng))
    return Sequential(layers)
