#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initialization Demo (Zeros, Large Random, He) for a 3-Layer Neural Network
--------------------------------------------------------------------------

This script reproduces the classic initialization experiment:
- Train the same 3-layer model with three different initializations.
- Visualize loss and decision boundaries.
- Compare outcomes (symmetry breaking, exploding/vanishing, He for ReLU).

Author: (your name)
License: MIT

Dependencies
------------
- numpy
- matplotlib
- sklearn (only for the provided dataset util, if available)
- init_utils.py        (provides: sigmoid, relu, compute_loss, forward_propagation,
                        backward_propagation, update_parameters, predict,
                        load_dataset, plot_decision_boundary, predict_dec)

Notes
-----
- This file does NOT rely on notebook magics.
- If `init_utils.py` is not present, the demo section will be skipped gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---- Optional utilities (from the original course helper) -------------------
# We import inside a try/except to keep the module importable on GitHub even if
# the helper file isn't present. The __main__ demo will check availability.
try:
    from init_utils import (  # type: ignore
        sigmoid,
        relu,
        compute_loss,
        forward_propagation,
        backward_propagation,
        update_parameters,
        predict,
        load_dataset,
        plot_decision_boundary,
        predict_dec,
    )
    _HAS_INIT_UTILS = True
except Exception:  # pragma: no cover - only for smoother GH experience
    _HAS_INIT_UTILS = False


# =============================================================================
# Initialization methods
# =============================================================================

def initialize_parameters_zeros(layers_dims: List[int]) -> Dict[str, np.ndarray]:
    """
    Initialize ALL weights and biases to zero.

    Parameters
    ----------
    layers_dims : list[int]
        Network layer sizes, including input layer.
        Example: [n_x, n_h1, n_h2, n_y]

    Returns
    -------
    parameters : dict
        Keys: "W1","b1",...,"WL","bL"
        Shapes:
            Wl: (layers_dims[l], layers_dims[l-1])
            bl: (layers_dims[l], 1)

    Warning
    -------
    This is intentionally *bad* for learning (no symmetry breaking).
    """
    parameters: Dict[str, np.ndarray] = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters[f"W{l}"] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters[f"b{l}"] = np.zeros((layers_dims[l], 1))
    return parameters


def initialize_parameters_random(layers_dims: List[int]) -> Dict[str, np.ndarray]:
    """
    Initialize weights to large random values (N(0,1) * 10) and biases to zero.

    Parameters
    ----------
    layers_dims : list[int]
        Network layer sizes, including input layer.

    Returns
    -------
    dict
        Parameter dictionary as in `initialize_parameters_zeros`.

    Notes
    -----
    This deliberately uses *large* random weights to illustrate why it can harm
    optimization (saturated sigmoid, exploding/vanishing gradients).
    """
    np.random.seed(3)  # reproducibility for the classic demo
    parameters: Dict[str, np.ndarray] = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10.0
        parameters[f"b{l}"] = np.zeros((layers_dims[l], 1))
    return parameters


def initialize_parameters_he(layers_dims: List[int]) -> Dict[str, np.ndarray]:
    """
    He initialization (recommended for ReLU family activations).

    Parameters
    ----------
    layers_dims : list[int]
        Network layer sizes, including input layer.

    Returns
    -------
    dict
        Parameter dictionary as in `initialize_parameters_zeros`.

    Formula
    -------
    W[l] ~ N(0, 2 / n_{l-1}), b[l] = 0
    """
    np.random.seed(3)  # reproducibility for comparison
    parameters: Dict[str, np.ndarray] = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters[f"W{l}"] = (
            np.random.randn(layers_dims[l], layers_dims[l - 1])
            * np.sqrt(2.0 / layers_dims[l - 1])
        )
        parameters[f"b{l}"] = np.zeros((layers_dims[l], 1))
    return parameters


# =============================================================================
# Training loop (uses helpers from init_utils)
# =============================================================================

@dataclass
class TrainConfig:
    learning_rate: float = 0.01
    num_iterations: int = 15000
    print_cost: bool = True
    initialization: str = "he"  # "zeros" | "random" | "he"


def model(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: TrainConfig = TrainConfig(),
) -> Dict[str, np.ndarray]:
    """
    Train a fixed 3-layer network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Parameters
    ----------
    X : np.ndarray
        Input of shape (n_x, m)
    Y : np.ndarray
        Labels of shape (1, m) ; 0 = red, 1 = blue
    cfg : TrainConfig
        Hyperparameters + initialization type

    Returns
    -------
    parameters : dict
        Learned parameters.
    """
    if not _HAS_INIT_UTILS:
        raise RuntimeError(
            "init_utils.py is required for forward/backward/compute_loss/predict."
        )

    layers_dims = [X.shape[0], 10, 5, 1]

    # --- Initialization -------------------------------------------------------
    if cfg.initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif cfg.initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif cfg.initialization == "he":
        parameters = initialize_parameters_he(layers_dims)
    else:
        raise ValueError(
            'initialization must be one of {"zeros", "random", "he"}'
        )

    # --- Gradient Descent -----------------------------------------------------
    costs: List[float] = []
    for i in range(cfg.num_iterations):
        # Forward pass
        a3, cache = forward_propagation(X, parameters)

        # Loss
        cost = compute_loss(a3, Y)

        # Backward pass
        grads = backward_propagation(X, Y, cache)

        # Update
        parameters = update_parameters(parameters, grads, cfg.learning_rate)

        # Log every 1000 iters (append cost regardless of print flag)
        if i % 1000 == 0:
            costs.append(cost)
            if cfg.print_cost:
                print(f"Cost after iteration {i}: {cost}")

    # --- Plot cost curve (x matches logging stride) ---------------------------
    if costs:
        xs = np.arange(0, cfg.num_iterations, 1000)
        plt.figure()
        plt.plot(xs, costs, marker="o")
        plt.ylabel("cost")
        plt.xlabel("iterations")
        plt.title(f"LR={cfg.learning_rate}, init={cfg.initialization}")
        plt.tight_layout()
        plt.show()

    return parameters


# =============================================================================
# Demo (runs only if init_utils is available)
# =============================================================================

def _run_demo() -> None:
    """
    Train the model with three initializations and visualize results.

    Skips gracefully if helper utilities are not available.
    """
    if not _HAS_INIT_UTILS:
        print(
            "[info] init_utils.py not found. "
            "Demo skipped. The module is otherwise importable."
        )
        return

    # Load the synthetic concentric-circles dataset from the helpers
    train_X, train_Y, test_X, test_Y = load_dataset()

    # --- ZEROS ---------------------------------------------------------------
    print("\n=== ZEROS initialization ===")
    params_zeros = model(
        train_X, train_Y,
        TrainConfig(initialization="zeros", print_cost=True)
    )
    print("On the train set:")
    _ = predict(train_X, train_Y, params_zeros)
    print("On the test set:")
    _ = predict(test_X, test_Y, params_zeros)

    plt.title("Decision Boundary — Zeros")
    ax = plt.gca()
    ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(params_zeros, x.T), train_X, train_Y)

    # --- LARGE RANDOM --------------------------------------------------------
    print("\n=== Large RANDOM initialization (x10) ===")
    params_rand = model(
        train_X, train_Y,
        TrainConfig(initialization="random", print_cost=True)
    )
    print("On the train set:")
    _ = predict(train_X, train_Y, params_rand)
    print("On the test set:")
    _ = predict(test_X, test_Y, params_rand)

    plt.title("Decision Boundary — Large Random")
    ax = plt.gca()
    ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(params_rand, x.T), train_X, train_Y)

    # --- HE ------------------------------------------------------------------
    print("\n=== HE initialization ===")
    params_he = model(
        train_X, train_Y,
        TrainConfig(initialization="he", print_cost=True)
    )
    print("On the train set:")
    _ = predict(train_X, train_Y, params_he)
    print("On the test set:")
    _ = predict(test_X, test_Y, params_he)

    plt.title("Decision Boundary — He")
    ax = plt.gca()
    ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(params_he, x.T), train_X, train_Y)


if __name__ == "__main__":
    _run_demo()
