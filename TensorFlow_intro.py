"""
Deep Learning TensorFlow Assignment
Author: Roy Naor
Description:
  Complete implementation of linear functions, activations,
  one-hot encoding, parameter initialization, forward propagation,
  and loss computation using TensorFlow.
"""

import tensorflow as tf
import numpy as np


# -------------------------------------------------------------
# 2.1 - Linear Function
# -------------------------------------------------------------
def linear_function():
    """
    Implements Y = W*X + b using random tensors.
    Shapes:
      W: (4,3), X: (3,1), b: (4,1)
    """
    np.random.seed(1)
    X = tf.constant(np.random.randn(3, 1), name="X")
    W = tf.constant(np.random.randn(4, 3), name="W")
    b = tf.constant(np.random.randn(4, 1), name="b")

    Y = tf.add(tf.matmul(W, X), b)
    return Y


# -------------------------------------------------------------
# 2.2 - Sigmoid Function
# -------------------------------------------------------------
def sigmoid(z):
    """
    Computes the sigmoid of z.
    Works for scalar or vector inputs.
    """
    z = tf.cast(z, dtype=tf.float32)
    a = tf.keras.activations.sigmoid(z)
    return a


# -------------------------------------------------------------
# 2.3 - One Hot Encoding
# -------------------------------------------------------------
def one_hot_matrix(label, C=6):
    """
    Returns one-hot encoded vector of length C.
    Example: label=2, C=4 -> [0,0,1,0]
    """
    one_hot = tf.one_hot(label, depth=C, axis=0)
    one_hot = tf.reshape(one_hot, shape=[C, ])
    return one_hot


# -------------------------------------------------------------
# 2.4 - Initialize Parameters
# -------------------------------------------------------------
def initialize_parameters():
    """
    Initializes weights and biases using GlorotNormal initializer.
    Shapes:
      W1: [25,12288], b1: [25,1]
      W2: [12,25],    b2: [12,1]
      W3: [6,12],     b3: [6,1]
    """
    initializer = tf.keras.initializers.GlorotNormal(seed=1)

    W1 = tf.Variable(initializer(shape=(25, 12288)), name="W1")
    b1 = tf.Variable(initializer(shape=(25, 1)), name="b1")

    W2 = tf.Variable(initializer(shape=(12, 25)), name="W2")
    b2 = tf.Variable(initializer(shape=(12, 1)), name="b2")

    W3 = tf.Variable(initializer(shape=(6, 12)), name="W3")
    b3 = tf.Variable(initializer(shape=(6, 1)), name="b3")

    parameters = {"W1": W1, "b1": b1,
                  "W2": W2, "b2": b2,
                  "W3": W3, "b3": b3}

    return parameters


# -------------------------------------------------------------
# 3.1 - Forward Propagation
# -------------------------------------------------------------
def forward_propagation(X, parameters):
    """
    Implements: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
    No softmax is applied here (handled in the loss).
    """
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    W3, b3 = parameters['W3'], parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.keras.activations.relu(Z1)

    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.keras.activations.relu(Z2)

    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3


# -------------------------------------------------------------
# 3.2 - Compute Total Loss
# -------------------------------------------------------------
def compute_total_loss(logits, labels):
    """
    Computes total categorical cross-entropy loss.
    Sums over all examples (not averaged).
    """
    total_loss = tf.reduce_sum(
        tf.keras.losses.categorical_crossentropy(
            y_true=tf.transpose(labels),
            y_pred=tf.transpose(logits),
            from_logits=True
        )
    )
    return total_loss


# -------------------------------------------------------------
# Example Run
# -------------------------------------------------------------
if __name__ == "__main__":
    print("✅ TensorFlow Assignment Ready")

    # Test linear function
    print("\nLinear Function Output:")
    print(linear_function())

    # Test sigmoid
    print("\nSigmoid(0) =", sigmoid(0).numpy())

    # Test one-hot
    print("\nOne-hot (label=2, C=5):", one_hot_matrix(2, C=5).numpy())

    # Test parameter init
    params = initialize_parameters()
    print("\nInitialized W1 shape:", params["W1"].shape)

    # Dummy forward prop + loss example
    X = tf.random.normal((12288, 10))
    logits = forward_propagation(X, params)
    labels = tf.one_hot(np.random.randint(0, 6, size=10), depth=6)
    labels = tf.transpose(labels)
    print("\nTotal loss:", compute_total_loss(logits, labels).numpy())
