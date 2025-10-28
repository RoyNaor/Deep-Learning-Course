"""
Optimization Algorithms for Deep Learning
-----------------------------------------
This module implements several gradient-based optimization techniques
commonly used for training neural networks, including:
    1. Batch Gradient Descent
    2. Mini-Batch Gradient Descent
    3. Momentum
    4. Adam Optimizer
    5. Learning Rate Decay (Exponential & Interval Scheduling)

Each function includes clean documentation and explanatory comments.
"""

import numpy as np
import math


# ============================================================
# 1. Gradient Descent
# ============================================================

def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using standard Gradient Descent.
    
    Arguments:
    parameters -- dictionary containing model parameters:
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    grads -- dictionary containing gradients for each parameter:
                    grads["dW" + str(l)] = dWl
                    grads["db" + str(l)] = dbl
    learning_rate -- learning rate (scalar)
    
    Returns:
    parameters -- dictionary with updated parameters
    """
    L = len(parameters) // 2  # number of layers

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters


# ============================================================
# 2. Mini-Batch Gradient Descent
# ============================================================

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates random mini-batches from (X, Y).
    
    Arguments:
    X -- input data of shape (input_size, number_of_examples)
    Y -- labels of shape (1, number_of_examples)
    mini_batch_size -- size of each mini-batch
    seed -- random seed for reproducibility
    
    Returns:
    mini_batches -- list of tuples (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition the data
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    # Handle the last mini-batch (if any)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches


# ============================================================
# 3. Momentum
# ============================================================

def initialize_velocity(parameters):
    """
    Initialize velocity as zero arrays matching parameter shapes.
    """
    L = len(parameters) // 2
    v = {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum.
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        # Compute velocity
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]

        # Update parameters
        parameters["W" + str(l)] -= learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * v["db" + str(l)]

    return parameters, v


# ============================================================
# 4. Adam Optimizer
# ============================================================

def initialize_adam(parameters):
    """
    Initialize v and s for the Adam optimizer.
    """
    L = len(parameters) // 2
    v, s = {}, {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam optimization.
    Combines Momentum and RMSProp.
    """
    L = len(parameters) // 2
    v_corrected, s_corrected = {}, {}

    for l in range(1, L + 1):
        # 1. Update biased first moment estimate
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]

        # 2. Compute bias-corrected first moment
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)

        # 3. Update biased second moment estimate
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)

        # 4. Compute bias-corrected second moment
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)

        # 5. Update parameters
        parameters["W" + str(l)] -= learning_rate * (
            v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        )
        parameters["b" + str(l)] -= learning_rate * (
            v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)
        )

    return parameters, v, s, v_corrected, s_corrected


# ============================================================
# 5. Learning Rate Decay
# ============================================================

def update_lr(learning_rate0, epoch_num, decay_rate):
    """
    Exponential learning rate decay.
    
    α = α0 / (1 + decay_rate * epoch_num)
    """
    learning_rate = learning_rate0 / (1 + decay_rate * epoch_num)
    return learning_rate


def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
    """
    Exponential learning rate decay with fixed interval scheduling.
    
    α = α0 / (1 + decay_rate * floor(epoch_num / time_interval))
    """
    learning_rate = learning_rate0 / (1 + decay_rate * np.floor(epoch_num / time_interval))
    return learning_rate


# ============================================================
# Example Usage
# ============================================================
if __name__ == "__main__":
    print("Optimization algorithms module loaded successfully.")
