#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Gradient Checking Implementation
# Author: Roy Naor
# Description: 1D and N-Dimensional gradient check for a 
# 3-layer neural network (LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID)
# ============================================================

import numpy as np

# ------------------------------------------------------------
# Forward Propagation (1D case)
# ------------------------------------------------------------
def forward_propagation(x, theta):
    """
    Implements the linear forward propagation (J(theta) = theta * x)
    Arguments:
        x -- real-valued input
        theta -- model parameter (scalar)
    Returns:
        J -- computed cost value
    """
    J = theta * x
    return J


# ------------------------------------------------------------
# Backward Propagation (1D case)
# ------------------------------------------------------------
def backward_propagation(x, theta):
    """
    Computes the derivative of J with respect to theta.
    dJ/dtheta = x
    Arguments:
        x -- real-valued input
        theta -- model parameter (scalar)
    Returns:
        dtheta -- gradient of the cost with respect to theta
    """
    dtheta = x
    return dtheta


# ------------------------------------------------------------
# Gradient Check (1D case)
# ------------------------------------------------------------
def gradient_check(x, theta, epsilon=1e-7, print_msg=False):
    """
    Implements gradient checking for the 1D case.
    Arguments:
        x -- input value
        theta -- parameter
        epsilon -- small value for finite differences
    Returns:
        difference -- difference between numerical and analytical gradients
    """
    theta_plus = theta + epsilon
    theta_minus = theta - epsilon
    J_plus = forward_propagation(x, theta_plus)
    J_minus = forward_propagation(x, theta_minus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)
    grad = backward_propagation(x, theta)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if print_msg:
        if difference > 2e-7:
            print(f"\033[93mThere is a mistake in the backward propagation! difference = {difference}\033[0m")
        else:
            print(f"\033[92mYour backward propagation works perfectly fine! difference = {difference}\033[0m")

    return difference


# ------------------------------------------------------------
# Forward Propagation (N-Dimensional network)
# ------------------------------------------------------------
def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def forward_propagation_n(X, Y, parameters):
    """
    Implements forward propagation and computes the logistic cost.
    Architecture: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Arguments:
        X -- input data (size: input_dim x m)
        Y -- true labels (1 x m)
        parameters -- dictionary containing W1, b1, W2, b2, W3, b3
    Returns:
        cost -- logistic cost
        cache -- all intermediate values
    """
    m = X.shape[1]
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    log_probs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1. / m * np.sum(log_probs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    return cost, cache


# ------------------------------------------------------------
# Backward Propagation (N-Dimensional network)
# ------------------------------------------------------------
def backward_propagation_n(X, Y, cache):
    """
    Implements backward propagation for the 3-layer neural network.
    Arguments:
        X -- input data
        Y -- true labels
        cache -- output of forward_propagation_n
    Returns:
        gradients -- dictionary containing all computed gradients
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return gradients


# ------------------------------------------------------------
# N-Dimensional Gradient Check
# ------------------------------------------------------------
def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7, print_msg=False):
    """
    Checks if backward_propagation_n computes the correct gradients.
    Arguments:
        parameters -- dictionary with model parameters
        gradients -- dictionary with gradients from backprop
        X -- input data
        Y -- true labels
        epsilon -- small value for finite differences
    Returns:
        difference -- relative difference between analytical and numerical gradients
    """
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]

    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    for i in range(num_parameters):
        theta_plus = np.copy(parameters_values)
        theta_plus[i] += epsilon
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))

        theta_minus = np.copy(parameters_values)
        theta_minus[i] -= epsilon
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))

        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if print_msg:
        if difference > 2e-7:
            print(f"\033[93mThere is a mistake in the backward propagation! difference = {difference}\033[0m")
        else:
            print(f"\033[92mYour backward propagation works perfectly fine! difference = {difference}\033[0m")

    return difference


# ------------------------------------------------------------
# Helper Functions (for vector reshaping)
# ------------------------------------------------------------
def dictionary_to_vector(parameters):
    """
    Converts parameter dictionary into a single vector.
    Returns:
        vector -- concatenated parameters
        keys -- list of parameter names
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenat
