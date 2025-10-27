#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regularization Demo (L2 & Dropout) for a 3-Layer Neural Network
----------------------------------------------------------------

This script trains the same model under three settings:
- No regularization (baseline)
- L2 regularization
- Dropout (inverted dropout on hidden layers)

It reproduces the Coursera DLS "Regularization" exercise in a script-friendly,
GitHub-ready format.

Author: (your name)
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Optional helpers from the course (reg_utils.py). The script stays importable
# even when helpers are missing; the demo is skipped gracefully.
# ---------------------------------------------------------------------------
try:
    from reg_utils import (  # type: ignore
        sigmoid, relu, plot_decision_boundary, initialize_parameters,
        load_2D_dataset, predict_dec, compute_cost, predict,
        forward_propagation, backward_propagation, update_parameters
    )
    _HAS_REG_UTILS = True
except Exception:  # pragma: no cover - allows the file to import without helpers
    _HAS_REG_UTILS = False


# =============================================================================
# L2 regularization
# =============================================================================

def compute_cost_with_regularization(A3: np.ndarray,
                                     Y: np.ndarray,
                                     parameters: Dict[str, np.ndarray],
                                     lambd: float) -> float:
    """
    Cross-entropy cost + L2 penalty (biases not regularized).

    J_reg = J + (λ / 2m) * (||W1||_F^2 + ||W2||_F^2 + ||W3||_F^2)
    """
    m = Y.shape[1]
    W1, W2, W3 = parameters["W1"], parameters["W2"], parameters["W3"]

    # base cross-entropy cost from helper
    ce_cost = compute_cost(A3, Y)

    # L2 penalty
    l2 = (lambd / (2.0 * m)) * (
        np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))
    )
    return float(ce_cost + l2)


def backward_propagation_with_regularization(X: np.ndarray,
                                             Y: np.ndarray,
                                             cache: Tuple[np.ndarray, ...],
                                             lambd: float) -> Dict[str, np.ndarray]:
    """
    Backprop for 3-layer net with L2 regularization.
    Only dW terms get + (λ/m) * W.
    """
    m = X.shape[1]
    (Z1, A1, W1, b1,
     Z2, A2, W2, b2,
     Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1. / m) * np.dot(dZ3, A2.T) + (lambd / m) * W3
    db3 = (1. / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * (A2 > 0).astype(np.int64)
    dW2 = (1. / m) * np.dot(dZ2, A1.T) + (lambd / m) * W2
    db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * (A1 > 0).astype(np.int64)
    dW1 = (1. / m) * np.dot(dZ1, X.T) + (lambd / m) * W1
    db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
            "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
            "dZ1": dZ1, "dW1": dW1, "db1": db1}


# =============================================================================
# Dropout (inverted dropout)
# =============================================================================

def forward_propagation_with_dropout(X: np.ndarray,
                                     parameters: Dict[str, np.ndarray],
                                     keep_prob: float = 0.5) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    Dropout applied to hidden layers using inverted dropout.
    """
    np.random.seed(1)

    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    # Layer 1
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = (np.random.rand(*A1.shape) < keep_prob).astype(int)
    A1 = (A1 * D1) / keep_prob

    # Layer 2
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = (np.random.rand(*A2.shape) < keep_prob).astype(int)
    A2 = (A2 * D2) / keep_prob

    # Output (no dropout)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1,
             Z2, D2, A2, W2, b2,
             Z3, A3, W3, b3)
    return A3, cache


def backward_propagation_with_dropout(X: np.ndarray,
                                      Y: np.ndarray,
                                      cache: Tuple[np.ndarray, ...],
                                      keep_prob: float) -> Dict[str, np.ndarray]:
    """
    Backprop for 3-layer net with inverted dropout on hidden layers.
    Reapply masks and scale by 1/keep_prob.
    """
    m = X.shape[1]
    (Z1, D1, A1, W1, b1,
     Z2, D2, A2, W2, b2,
     Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1. / m) * np.dot(dZ3, A2.T)
    db3 = (1. / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dA2 = (dA2 * D2) / keep_prob
    dZ2 = dA2 * (A2 > 0).astype(np.int64)
    dW2 = (1. / m) * np.dot(dZ2, A1.T)
    db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = (dA1 * D1) / keep_prob
    dZ1 = dA1 * (A1 > 0).astype(np.int64)
    dW1 = (1. / m) * np.dot(dZ1, X.T)
    db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
            "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
            "dZ1": dZ1, "dW1": dW1, "db1": db1}


# =============================================================================
# Training loop (shared by all settings)
# =============================================================================

@dataclass
class TrainConfig:
    learning_rate: float = 0.3
    num_iterations: int = 30000
    print_cost: bool = True
    lambd: float = 0.0         # L2 strength; 0 disables L2
    keep_prob: float = 1.0     # 1 disables dropout


def model(X: np.ndarray, Y: np.ndarray, cfg: TrainConfig = TrainConfig()
          ) -> Dict[str, np.ndarray]:
    """
    Train a fixed 3-layer network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    L2 and Dropout are mutually exclusive in this demo (as in the course).
    """
    if not _HAS_REG_UTILS:
        raise RuntimeError("reg_utils.py is required for this demo (forward/backward/update).")

    assert (cfg.lambd == 0.0 or cfg.keep_prob == 1.0), \
        "This demo toggles one technique at a time (either L2 or Dropout)."

    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]
    parameters = initialize_parameters(layers_dims)

    costs: List[float] = []

    for i in range(cfg.num_iterations):
        # Forward
        if cfg.keep_prob == 1.0:
            A3, cache = forward_propagation(X, parameters)
        else:
            A3, cache = forward_propagation_with_dropout(X, parameters, cfg.keep_prob)

        # Cost
        if cfg.lambd == 0.0:
            cost = compute_cost(A3, Y)
        else:
            cost = compute_cost_with_regularization(A3, Y, parameters, cfg.lambd)

        # Backward
        if cfg.lambd == 0.0 and cfg.keep_prob == 1.0:
            grads = backward_propagation(X, Y, cache)
        elif cfg.lambd != 0.0:
            grads = backward_propagation_with_regularization(X, Y, cache, cfg.lambd)
        else:
            grads = backward_propagation_with_dropout(X, Y, cache, cfg.keep_prob)

        # Update
        parameters = update_parameters(parameters, grads, cfg.learning_rate)

        # Log every 1000 steps
        if i % 1000 == 0:
            costs.append(cost)
            if cfg.print_cost and i % 10000 == 0:
                print(f"Cost after iteration {i}: {cost}")

    # Plot cost curve
    if costs:
        xs = np.arange(0, cfg.num_iterations, 1000)
        plt.figure()
        plt.plot(xs, costs, marker="o")
        plt.ylabel("cost")
        plt.xlabel("iterations")
        title_bits = []
        if cfg.lambd != 0.0: title_bits.append(f"L2={cfg.lambd}")
        if cfg.keep_prob != 1.0: title_bits.append(f"dropout p={cfg.keep_prob}")
        title_bits.append(f"LR={cfg.learning_rate}")
        plt.title(", ".join(title_bits))
        plt.tight_layout()
        plt.show()

    return parameters


# =============================================================================
# Demo (only runs if helpers are available)
# =============================================================================

def _run_demo() -> None:
    if not _HAS_REG_UTILS:
        print("[info] reg_utils.py not found. Demo skipped.")
        return

    # Data
    train_X, train_Y, test_X, test_Y = load_2D_dataset()

    # --- Baseline -------------------------------------------------------------
    print("\n=== Baseline (no regularization) ===")
    params = model(train_X, train_Y, TrainConfig(keep_prob=1.0, lambd=0.0))
    print("On the training set:"); _ = predict(train_X, train_Y, params)
    print("On the test set:");     _ = predict(test_X, test_Y, params)
    plt.title("Decision Boundary — Baseline")
    ax = plt.gca(); ax.set_xlim([-0.75, 0.40]); ax.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, train_Y)

    # --- L2 -------------------------------------------------------------------
    print("\n=== L2 regularization (λ=0.7) ===")
    params_l2 = model(train_X, train_Y, TrainConfig(lambd=0.7, keep_prob=1.0))
    print("On the training set:"); _ = predict(train_X, train_Y, params_l2)
    print("On the test set:");     _ = predict(test_X, test_Y, params_l2)
    plt.title("Decision Boundary — L2")
    ax = plt.gca(); ax.set_xlim([-0.75, 0.40]); ax.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(params_l2, x.T), train_X, train_Y)

    # --- Dropout --------------------------------------------------------------
    print("\n=== Dropout (keep_prob=0.86) ===")
    params_do = model(train_X, train_Y, TrainConfig(keep_prob=0.86, lambd=0.0, learning_rate=0.3))
    print("On the training set:"); _ = predict(train_X, train_Y, params_do)
    print("On the test set:");     _ = predict(test_X, test_Y, params_do)
    plt.title("Decision Boundary — Dropout")
    ax = plt.gca(); ax.set_xlim([-0.75, 0.40]); ax.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(params_do, x.T), train_X, train_Y)


if __name__ == "__main__":
    _run_demo()
