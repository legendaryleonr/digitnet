# How DigitNet works (plain-English notes)

## 1. The data

Each image is 8×8 = **64 numbers**, one per pixel, from 0 (black) to 1 (white ink). The label is the digit 0–9. There are 1,797 images, split into train (1,221), validation (216), and test (360). The test set is only used once, at the very end.

## 2. The forward pass (making a guess)

- **Dense layer:** `output = input @ W + b`. Every output is a weighted sum of every input. `W` and `b` are the numbers the network *learns*.
- **ReLU:** `max(0, x)`. Without a non-linearity, stacking Dense layers would collapse into one big linear function, and the network couldn't learn curvy decision boundaries.
- **Softmax:** turns the final 10 scores into probabilities that add up to 1. We subtract the largest score first so `exp()` doesn't overflow.

## 3. The loss (how wrong was the guess)

**Cross-entropy** = `−log(probability given to the correct digit)`. If the net is 99% sure of the right answer, the loss is about 0.01. If it gives the right answer 1%, the loss is about 4.6. It punishes confident wrong answers heavily.

## 4. Backpropagation (who is to blame)

Using the chain rule, each layer's `backward()` takes "how much did my output affect the loss" and computes: - how much each of its **weights** affected the loss (used to update them), and - how much its **input** affected the loss (passed back to the previous layer).

Key results: - Dense: `dW = xᵀ · grad`, `db = sum(grad)`, `dx = grad · Wᵀ` - ReLU: pass the gradient through where the input was \> 0, otherwise 0 - Softmax + cross-entropy together: `grad = (probs − one_hot(y)) / N`, which is a nice simplification

## 5. Gradient checking (is the calculus right?)

For each weight, nudge it up and down by a tiny ε and measure how the loss changes: `(L(w+ε) − L(w−ε)) / 2ε`. This is slow but obviously correct. The tests require backprop to agree with it to about 1 part in a million.

## 6. Optimizers (how to update)

- **SGD + momentum:** step downhill, but keep some "velocity" from previous steps so it doesn't zig-zag.
- **Adam:** keeps a running average of the gradient (momentum) and of the squared gradient, so each weight gets its own adaptive step size. Bias correction fixes the fact that both averages start at zero. This is what `train.py` uses.

## 7. Fighting overfitting

- **Dropout:** randomly zero 20% of hidden activations during training so the net can't rely on any single neuron. It's turned off at test time ("inverted" dropout scales during training so nothing changes at inference).
- **Weight decay:** a small pull of every weight toward zero.
- **Augmentation:** randomly shift half the training images by one pixel. This was the biggest single improvement.
- **Best checkpoint:** keep the weights from the epoch with the best validation accuracy.

## 8. The demo

`train.py` exports `W` and `b` for each Dense layer to `weights.json`. The web page redoes the forward pass in JavaScript: matrix multiply, ReLU, and softmax. Dropout isn't needed at inference time.
