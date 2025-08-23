#!/usr/bin/env python3
# Logistic Regression (Cats vs Non-Cats)

import argparse
import copy
import os
from typing import Dict, Tuple, List

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ----------------------------- Data -----------------------------

def load_dataset(
    train_path: str = "datasets/train_catvnoncat.h5",
    test_path: str = "datasets/test_catvnoncat.h5",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load HDF5 cat/non-cat dataset (same format as the Coursera utilities)."""
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Missing dataset files.\nExpected:\n  {train_path}\n  {test_path}"
        )
    with h5py.File(train_path, "r") as train_h5:
        train_x = np.array(train_h5["train_set_x"][:])          # (m_train, 64,64,3)
        train_y = np.array(train_h5["train_set_y"][:])          # (m_train,)
        classes = np.array(train_h5["list_classes"][:])         # (2,)

    with h5py.File(test_path, "r") as test_h5:
        test_x = np.array(test_h5["test_set_x"][:])             # (m_test, 64,64,3)
        test_y = np.array(test_h5["test_set_y"][:])             # (m_test,)

    return train_x, train_y.reshape(1, -1), test_x, test_y.reshape(1, -1), classes


# --------------------------- Model Core --------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def initialize_with_zeros(dim: int) -> Tuple[np.ndarray, float]:
    return np.zeros((dim, 1)), 0.0

def propagate(
    w: np.ndarray, b: float, X: np.ndarray, Y: np.ndarray
) -> Tuple[Dict[str, np.ndarray], float]:
    m = X.shape[1]
    Z = np.dot(w.T, X) + b          # (1,m)
    A = sigmoid(Z)                  # (1,m)
    eps = 1e-15                     # numerical safety
    cost = -(1/m) * np.sum(Y*np.log(A+eps) + (1-Y)*np.log(1-A+eps))
    dZ = A - Y
    dw = (1/m) * np.dot(X, dZ.T)    # (n_x,1)
    db = (1/m) * np.sum(dZ)         # scalar
    return {"dw": dw, "db": float(db)}, float(np.squeeze(cost))

def optimize(
    w: np.ndarray,
    b: float,
    X: np.ndarray,
    Y: np.ndarray,
    num_iterations: int = 2000,
    learning_rate: float = 0.5,
    print_cost: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[float]]:
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs: List[float] = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        w -= learning_rate * grads["dw"]
        b -= learning_rate * grads["db"]
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after {i:4d}: {cost:.6f}")

    return {"w": w, "b": b}, grads, costs

def predict(w: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    A = sigmoid(np.dot(w.T, X) + b)
    return (A > 0.5).astype(int)

def model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    num_iterations: int = 2000,
    learning_rate: float = 0.5,
    print_cost: bool = False,
) -> Dict[str, np.ndarray]:
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w, b = params["w"], params["b"]
    Y_pred_test = predict(w, b, X_test)
    Y_pred_train = predict(w, b, X_train)

    if print_cost:
        train_acc = 100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100
        test_acc  = 100 - np.mean(np.abs(Y_pred_test  - Y_test )) * 100
        print(f"train accuracy: {train_acc:.2f} %")
        print(f"test  accuracy: {test_acc :.2f} %")

    return {
        "costs": np.array(costs),
        "Y_prediction_test": Y_pred_test,
        "Y_prediction_train": Y_pred_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    }


# --------------------------- Utilities ---------------------------

def flatten_standardize(X: np.ndarray) -> np.ndarray:
    X_flat = X.reshape(X.shape[0], -1).T  # (n_x, m)
    return X_flat / 255.0

def plot_costs(costs: np.ndarray, lr: float, title: str = "Learning curve") -> None:
    if costs.size == 0:
        return
    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title(f"{title} — lr={lr}")
    plt.tight_layout()
    plt.show()

def predict_image(img_path: str, w: np.ndarray, b: float, num_px: int = 64) -> int:
    img = np.array(Image.open(img_path).resize((num_px, num_px)))
    x = (img.reshape(1, -1).T) / 255.0
    return int(np.squeeze(predict(w, b, x)))


# ------------------------------ CLI ------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Logistic Regression (Cats vs Non-Cats)")
    p.add_argument("--train", default="datasets/train_catvnoncat.h5")
    p.add_argument("--test",  default="datasets/test_catvnoncat.h5")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--image", help="Path to a custom image to classify")
    args = p.parse_args()

    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset(args.train, args.test)

    num_px = train_x_orig.shape[1]
    X_train = flatten_standardize(train_x_orig)
    X_test  = flatten_standardize(test_x_orig)

    print(f"train: {X_train.shape}, test: {X_test.shape}")

    out = model(
        X_train, train_y, X_test, test_y,
        num_iterations=args.iters, learning_rate=args.lr, print_cost=True
    )

    if args.plot:
        plot_costs(out["costs"], out["learning_rate"])

    if args.image:
        label = predict_image(args.image, out["w"], out["b"], num_px=num_px)
        name = classes[label].decode("utf-8") if hasattr(classes[label], "decode") else str(classes[label])
        print(f"Prediction for {os.path.basename(args.image)}: {label} ({name})")


if __name__ == "__main__":
    main()
