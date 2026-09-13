# dlp-assignment-01
# DLP Assignment 01 — 23F-0877

A PyTorch/NumPy notebook exploring the mechanics of multi-layer perceptrons (MLPs) on the **Fashion-MNIST** dataset (with one detour into regression on the California Housing dataset), moving from a from-scratch NumPy implementation up through activation functions, loss functions, optimizers, overfitting, regularization, and finally hyperparameter tuning.

## Dataset

- **Fashion-MNIST** (`zalando-research/fashionmnist`, loaded from Kaggle's `/kaggle/input`): 10-class grayscale clothing images, flattened to 784-dim vectors and normalized to `[0, 1]`.
- Split into **48,000 train / 12,000 validation / 10,000 test** samples (80/20 stratified split of the training set, fixed `test.csv` held out separately).
- **California Housing** (`sklearn.datasets.fetch_california_housing`): used only in Part 3 as a regression task, separate from the Fashion-MNIST classification work.

## Environment

- Python 3.12, run on Kaggle with GPU acceleration (2 GPUs detected in the environment).
- Libraries: `numpy`, `pandas`, `torch`, `torch.nn` / `torch.nn.functional`, `scikit-learn`, `matplotlib`, `seaborn`, `torchvision.transforms`.

## Structure

The notebook is organized into seven parts, each building on the last.

### Part 1 — MLP from scratch (NumPy) vs. PyTorch autograd
- Implements a 2-layer MLP (`784 → 64 → 10`) entirely in NumPy: He-initialized weights, manual `forward`/`backward` passes, ReLU + softmax, cross-entropy loss.
- Cross-checks the manually derived gradients against PyTorch's autograd on an equivalent `nn.Module`, confirming the manual backprop implementation is correct.

### Part 2 — Activation functions and vanishing gradients
- Trains the same architecture with four activations — **sigmoid, tanh, ReLU, leaky ReLU** — for 15 epochs each.
- Tracks the mean absolute gradient flowing into the first hidden layer per activation to compare how prone each is to vanishing gradients.

### Part 3 — Loss functions, classification vs. regression
- Compares **cross-entropy vs. MSE** as the loss for the classification task, training identical models otherwise.
- Extends into a small **regression MLP** (`8 → 32 → 16 → 1`) trained on California Housing with Adam, as a contrasting non-classification use case.
- **Finding:** cross-entropy converges faster because its gradient scales directly with the prediction error, whereas MSE's gradient is damped by the softmax derivative once predictions are confident — producing the near-flat MSE loss curve observed.

### Part 4 — Optimizer comparison
- Benchmarks **SGD, SGD+momentum, RMSProp, and Adam**, both at a shared learning rate and after per-optimizer learning-rate tuning, tracking epochs-to-85%-val-accuracy, final validation accuracy, and wall-clock time.
- **Finding:** Adam reached 85% validation accuracy fastest (2 epochs) and gave the best final accuracy (89.11%) with minimal tuning, at only ~2 seconds extra wall-clock cost over 20 epochs versus plain SGD.

### Part 5 — Deliberately inducing overfitting
- Trains an oversized model (`BigMLP`: `784 → 512 → 512 → 512 → 512 → 10`) on a small data subset for many extra epochs.
- Diagnoses the result as **high variance**: 99.35% train accuracy vs. a 17.23-point train/validation gap, with validation loss climbing while training loss keeps falling — ruling out high bias.

### Part 6 — Regularization techniques
- Applies and compares, all against the Part 5 baseline: **L2 weight decay** (λ = 1e-4, 1e-3, 1e-2), **L1 penalty**, **dropout** (rates 0.2 / 0.5 / 0.7), **batch normalization**, **early stopping** (patience = 5), **data augmentation** (random horizontal flip + rotation via `torchvision.transforms`), and simply **more training data** (10k / 20k samples).
- Reports train/validation accuracy and the generalization gap for each method in a comparison table.
- **Finding:** early stopping gave the best overall trade-off (cut the gap from 17.23 to 6.26 points in only 13 epochs), with dropout at 0.5 achieving an even smaller gap (4.60 points) at the cost of needing to search over dropout rates first.

### Part 7 — Hyperparameter tuning and final evaluation
- Defines a configurable `TunableMLP` (variable hidden width, optional dropout) and searches over hyperparameter configurations using **5-fold cross-validation** on a data subset.
- Trains the best configuration as a final model (with early stopping) on the full training set and evaluates it on the held-out test set with accuracy, macro precision/recall/F1, and a confusion matrix.
- **Finding:** the tuned model reached **88.16% test accuracy**, a 4.81-point improvement over the Part 2 baseline (83.35%), driven by a wider hidden layer (256 vs. 128 units), a cross-validated learning rate, and early stopping. Remaining errors cluster among visually similar classes (T-shirts, pullovers, coats, shirts), which is expected rather than a modeling flaw.

## How to run

1. Open in a Kaggle notebook environment (or adapt the `/kaggle/input/...` paths to a local copy of the Fashion-MNIST CSVs).
2. Run cells top to bottom — later parts depend on variables and helper functions (`X_train`, `X_val`, `device`, etc.) defined earlier in the notebook.
3. A GPU is recommended; the notebook checks for and uses one automatically via `torch.device`.

## Key takeaways

| Topic | Best choice found | Why |
|---|---|---|
| Loss (classification) | Cross-entropy | Faster convergence than MSE — gradient doesn't vanish as predictions sharpen |
| Optimizer | Adam | Fastest convergence, best final accuracy, negligible extra cost |
| Regularization | Early stopping (or dropout 0.5) | Best gap reduction for the accuracy trade-off |
| Final tuned model | 256 hidden units, CV-tuned LR, early stopping | 88.16% test accuracy (+4.81 pts over baseline) |
