"""Train the from-scratch network on the scikit-learn digits dataset.

Usage:
    python train.py            # train, evaluate, save plots + demo weights
    python train.py --epochs 5 # quick run
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from digitnet import Adam, SoftmaxCrossEntropy, build_mlp  # noqa: E402


def augment(x, rng):
    """Randomly shift each 8x8 image by up to 1 pixel. Helps the demo handle
    digits that aren't perfectly centered."""
    imgs = x.reshape(-1, 8, 8)
    out = np.zeros_like(imgs)
    for i, img in enumerate(imgs):
        if rng.random() < 0.5:  # leave half the images untouched
            out[i] = img
            continue
        dy, dx = rng.integers(-1, 2, size=2)
        shifted = np.zeros_like(img)  # shift with zero padding (no wrap-around)
        ys, yd = (slice(0, 8 - dy), slice(dy, 8)) if dy >= 0 else (slice(-dy, 8), slice(0, 8 + dy))
        xs, xd = (slice(0, 8 - dx), slice(dx, 8)) if dx >= 0 else (slice(-dx, 8), slice(0, 8 + dx))
        shifted[yd, xd] = img[ys, xs]
        out[i] = shifted
    return out.reshape(-1, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("demo", exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # pixel values are 0..16, scale to 0..1
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=args.seed)

    model = build_mlp(seed=args.seed)
    loss_fn = SoftmaxCrossEntropy()
    opt = Adam(model.params(), lr=args.lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_acc": []}
    best_val, best_weights = -1, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        idx = rng.permutation(len(X_tr))
        Xa = augment(X_tr[idx], rng)
        ya = y_tr[idx]
        losses = []
        for s in range(0, len(Xa), args.batch_size):
            xb, yb = Xa[s:s + args.batch_size], ya[s:s + args.batch_size]
            losses.append(loss_fn.forward(model.forward(xb, training=True), yb))
            model.backward(loss_fn.backward())
            opt.step()
        val_acc = (model.predict(X_val) == y_val).mean()
        history["train_loss"].append(float(np.mean(losses)))
        history["val_acc"].append(float(val_acc))
        if val_acc > best_val:  # keep the best checkpoint (early-stopping style)
            best_val = val_acc
            best_weights = [p.copy() for p, _ in model.params()]
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  val_acc {val_acc:.4f}")
    train_time = time.time() - t0

    for (p, _), w in zip(model.params(), best_weights):
        p[...] = w
    test_acc = float((model.predict(X_te) == y_te).mean())

    # baselines, so the number means something
    logreg = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    sk_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=args.seed).fit(X_tr, y_tr)
    metrics = {
        "digitnet_test_acc": round(test_acc, 4),
        "logistic_regression_test_acc": round(float(logreg.score(X_te, y_te)), 4),
        "sklearn_mlp_test_acc": round(float(sk_mlp.score(X_te, y_te)), 4),
        "best_val_acc": round(float(best_val), 4),
        "epochs": args.epochs,
        "train_seconds": round(train_time, 1),
        "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_te),
    }
    print(json.dumps(metrics, indent=2))
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(history["train_loss"]); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(history["val_acc"]); ax[1].set_title("Validation accuracy"); ax[1].set_xlabel("epoch")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("results/training_curves.png", dpi=140); plt.close(fig)

    cm = confusion_matrix(y_te, model.predict(X_te))
    fig, a = plt.subplots(figsize=(5, 4.5))
    im = a.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            a.text(j, i, cm[i, j], ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    a.set_xticks(range(10)); a.set_yticks(range(10))
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title("Confusion matrix (test set)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig("results/confusion_matrix.png", dpi=140); plt.close(fig)

    model.export_json("demo/weights.json", meta={"test_acc": test_acc, "input": "8x8 grayscale, 0..1"})
    print("saved results/ and demo/weights.json")


if __name__ == "__main__":
    main()
