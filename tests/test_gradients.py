"""Gradient checks: compare backprop gradients with numerical finite differences.

If backprop is implemented correctly, both numbers should agree to ~1e-6.
This is the main way to know the math in layers.py is right.
"""
import numpy as np

from digitnet import Dense, ReLU, Sequential, SoftmaxCrossEntropy, softmax


def numerical_grad(f, p, eps=1e-5):
    g = np.zeros_like(p)
    it = np.nditer(p, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = p[i]
        p[i] = old + eps; up = f()
        p[i] = old - eps; down = f()
        p[i] = old
        g[i] = (up - down) / (2 * eps)
        it.iternext()
    return g


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def test_network_gradients_match_numerical():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(5, 6))
    y = rng.integers(0, 3, size=5)
    model = Sequential([Dense(6, 8, rng), ReLU(), Dense(8, 3, rng)])
    loss_fn = SoftmaxCrossEntropy()

    def loss():
        return loss_fn.forward(model.forward(x), y)

    loss()
    model.backward(loss_fn.backward())
    for p, g in model.params():
        assert rel_error(g, numerical_grad(loss, p)) < 1e-6


def test_input_gradient_matches_numerical():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 5))
    y = rng.integers(0, 4, size=4)
    model = Sequential([Dense(5, 7, rng), ReLU(), Dense(7, 4, rng)])
    loss_fn = SoftmaxCrossEntropy()

    def loss():
        return loss_fn.forward(model.forward(x), y)

    loss()
    dx = model.backward(loss_fn.backward())
    assert rel_error(dx, numerical_grad(loss, x)) < 1e-6


def test_softmax_is_stable_and_sums_to_one():
    p = softmax(np.array([[1000.0, 1000.0, -1000.0]]))
    assert np.all(np.isfinite(p))
    assert np.isclose(p.sum(), 1.0)


def test_can_overfit_tiny_batch():
    """Sanity check: a working network should memorize 10 examples."""
    from digitnet import Adam
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 8))
    y = np.arange(10) % 3
    model = Sequential([Dense(8, 32, rng), ReLU(), Dense(32, 3, rng)])
    loss_fn, opt = SoftmaxCrossEntropy(), Adam(model.params(), lr=1e-2)
    for _ in range(300):
        loss_fn.forward(model.forward(x), y)
        model.backward(loss_fn.backward())
        opt.step()
    assert (model.predict(x) == y).all()
