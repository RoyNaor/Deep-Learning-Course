#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset


# -------- Core NN building blocks --------

def initialize_parameters(n_x: int, n_h: int, n_y: int):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward_propagation(X, parameters):
    W1, b1, W2, b2 = (
        parameters["W1"],
        parameters["b1"],
        parameters["W2"],
        parameters["b2"],
    )
    Z1 = W1 @ X + b1
    A1 = np.tanh(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def compute_cost(A2, Y):
    m = Y.shape[1]
    A2 = np.clip(A2, 1e-15, 1 - 1e-15)  # numeric safety
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return float(cost)


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W2 = parameters["W2"]
    A1, A2 = cache["A1"], cache["A2"]

    dZ2 = A2 - Y
    dW2 = (dZ2 @ A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = (W2.T @ dZ2) * (1 - A1**2)
    dW1 = (dZ1 @ X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_parameters(parameters, grads, learning_rate=1.2):
    # copy to avoid in-place side effects
    params = {k: v.copy() for k, v in parameters.items()}
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params


def nn_model(
    X,
    Y,
    n_h=4,
    num_iterations=10000,
    learning_rate=1.2,
    print_cost=False,
    seed=3,
):
    np.random.seed(seed)
    n_x, n_y = X.shape[0], Y.shape[0]
    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost:.6f}")
            costs.append(cost)

    return parameters, costs


def predict(parameters, X):
    A2, _ = forward_propagation(X, parameters)
    return (A2 > 0.5).astype(int)


# -------- Train & evaluate --------

def train_and_plot(n_h=4, iters=10000, lr=1.2, seed=3, show_plot=True):
    X, Y = load_planar_dataset()
    parameters, _ = nn_model(
        X,
        Y,
        n_h=n_h,
        num_iterations=iters,
        learning_rate=lr,
        print_cost=True,
        seed=seed,
    )

    preds = predict(parameters, X)
    acc = float(
        (np.dot(Y, preds.T) + np.dot(1 - Y, 1 - preds.T)) / Y.size * 100
    )
    print(f"Accuracy: {acc:.1f}%")

    if show_plot:
        plt.figure()
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        plt.title(f"Decision Boundary (n_h={n_h})")
        plt.show()

    return parameters


if __name__ == "__main__":
    # tweak n_h / iters / lr if you like
    train_and_plot(n_h=4, iters=10000, lr=1.2, seed=3)
