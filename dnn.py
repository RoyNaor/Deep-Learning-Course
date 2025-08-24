"""
Minimal L-layer fully-connected neural network (binary classification).
Implements initialization, forward pass, cost, backward pass, parameter update,
prediction, and a simple training loop.

Author: you :)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

# -------------------------- Activations -------------------------- #

def sigmoid(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sigmoid activation."""
    A = 1.0 / (1.0 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ReLU activation."""
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def sigmoid_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """Backprop through a sigmoid unit."""
    Z = cache
    s = 1.0 / (1.0 + np.exp(-Z))
    dZ = dA * s * (1.0 - s)
    return dZ


def relu_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """Backprop through a ReLU unit."""
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


# ---------------------- Parameters initialization ---------------------- #

def initialize_parameters(n_x: int, n_h: int, n_y: int, seed: int | None = 1) -> Dict[str, np.ndarray]:
    """
    2-layer network convenience initializer.
    Shapes: W1(n_h, n_x), b1(n_h,1), W2(n_y, n_h), b2(n_y,1)
    """
    if seed is not None:
        np.random.seed(seed)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def initialize_parameters_deep(layer_dims: List[int], seed: int | None = 3) -> Dict[str, np.ndarray]:
    """
    L-layer network initializer.
    layer_dims: [n0, n1, ..., nL]
    Returns dict with keys W1..WL, b1..bL
    """
    if seed is not None:
        np.random.seed(seed)

    parameters: Dict[str, np.ndarray] = {}
    L = len(layer_dims)  # number of layers including input

    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
        # quick sanity checks
        assert parameters[f"W{l}"].shape == (layer_dims[l], layer_dims[l - 1])
        assert parameters[f"b{l}"].shape == (layer_dims[l], 1)

    return parameters


# ------------------------------ Forward pass ------------------------------ #

def linear_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute Z = W A_prev + b."""
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache


def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str
                              ) -> Tuple[np.ndarray, Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]:
    """
    Forward step for [LINEAR -> ACTIVATION].
    activation: "relu" or "sigmoid"
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    else:
        raise ValueError("activation must be 'relu' or 'sigmoid'")
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X: np.ndarray, parameters: Dict[str, np.ndarray]
                    ) -> Tuple[np.ndarray, List[Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]]:
    """
    Implements [LINEAR->RELU] * (L-1) -> [LINEAR->SIGMOID].
    Returns AL and list of caches for backprop.
    """
    caches: List[Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = []
    A = X
    L = len(parameters) // 2  # only weight/bias layers

    # Hidden layers: ReLU
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters[f"W{l}"], parameters[f"b{l}"], activation="relu")
        caches.append(cache)

    # Output layer: Sigmoid
    AL, cache = linear_activation_forward(A, parameters[f"W{L}"], parameters[f"b{L}"], activation="sigmoid")
    caches.append(cache)

    return AL, caches


# --------------------------------- Cost --------------------------------- #

def compute_cost(AL: np.ndarray, Y: np.ndarray, eps: float = 1e-15) -> float:
    """
    Binary cross-entropy cost:
    J = -(1/m) sum( y log(AL) + (1-y) log(1-AL) )
    """
    m = Y.shape[1]
    # numerical stability
    AL = np.clip(AL, eps, 1.0 - eps)
    cost = (1.0 / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
    return float(np.squeeze(cost))


# ------------------------------ Backward pass ------------------------------ #

def linear_backward(dZ: np.ndarray, cache: Tuple[np.ndarray, np.ndarray, np.ndarray]
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward step for the linear part."""
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1.0 / m) * np.dot(dZ, A_prev.T)
    db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA: np.ndarray,
                               cache: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                               activation: str
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward step for [ACTIVATION -> LINEAR].
    activation: "relu" or "sigmoid"
    """
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        raise ValueError("activation must be 'relu' or 'sigmoid'")
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL: np.ndarray, Y: np.ndarray,
                     caches: List[Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]
                     ) -> Dict[str, np.ndarray]:
    """
    Backprop for [LINEAR->RELU] * (L-1) -> [LINEAR->SIGMOID].
    Returns grads dict with dW1..dWL, db1..dbL, and dA0..dA(L-1).
    """
    grads: Dict[str, np.ndarray] = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    # dAL from dJ/dAL for BCE
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Last layer (sigmoid)
    current_cache = caches[L - 1]
    dA_prev, dW, db = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    grads[f"dA{L-1}"] = dA_prev
    grads[f"dW{L}"] = dW
    grads[f"db{L}"] = db

    # Loop over remaining layers (ReLU)
    for l in reversed(range(L - 1)):  # L-2 .. 0
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads[f"dA{l+1}"], current_cache, activation="relu")
        grads[f"dA{l}"] = dA_prev
        grads[f"dW{l+1}"] = dW
        grads[f"db{l+1}"] = db

    return grads


# ------------------------------ Update step ------------------------------ #

def update_parameters(parameters: Dict[str, np.ndarray],
                      grads: Dict[str, np.ndarray],
                      learning_rate: float) -> Dict[str, np.ndarray]:
    """Gradient descent update for every Wl and bl."""
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] = parameters[f"W{l}"] - learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] = parameters[f"b{l}"] - learning_rate * grads[f"db{l}"]
    return parameters


# ------------------------------ Utilities ------------------------------ #

def predict(X: np.ndarray, parameters: Dict[str, np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Return binary predictions (0/1) for X."""
    AL, _ = L_model_forward(X, parameters)
    return (AL >= threshold).astype(int)


def L_layer_model(X: np.ndarray, Y: np.ndarray, layers_dims: List[int],
                  learning_rate: float = 0.0075, num_iterations: int = 3000,
                  print_cost: bool = False, seed: int | None = 3
                  ) -> Dict[str, np.ndarray]:
    """
    Train an L-layer model with gradient descent.
    """
    parameters = initialize_parameters_deep(layers_dims, seed=seed)

    for i in range(num_iterations):
        # Forward
        AL, caches = L_model_forward(X, parameters)
        # Cost (optional)
        cost = compute_cost(AL, Y)
        # Backward
        grads = L_model_backward(AL, Y, caches)
        # Update
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print(f"Iteration {i:5d} | cost = {cost:.6f}")

    return parameters


# ------------------------------ Example run ------------------------------ #

if __name__ == "__main__":
    # Tiny demonstration on synthetic data (not meant for accuracy).
    np.random.seed(0)
    m = 400
    n_x = 2
    X = np.random.randn(n_x, m)

    # Non-linear boundary labels
    Y = (np.sin(X[0, :] * 2.0) + 0.5 * X[1, :] > 0).astype(int).reshape(1, m)

    layers = [n_x, 5, 3, 1]
    params = L_layer_model(X, Y, layers, learning_rate=0.05, num_iterations=1500, print_cost=True, seed=1)

    preds = predict(X, params)
    acc = float(np.mean(preds == Y))
    print(f"Train accuracy: {acc * 100:.2f}%")
