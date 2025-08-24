# deep_nn_app.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dnn import *

np.random.seed(1)

def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []
    n_x, n_h, n_y = layers_dims
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
    for i in range(num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        cost = compute_cost(A2, Y)
        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        _, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        parameters = update_parameters(parameters, grads, learning_rate)
        W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print(f"Cost after iteration {i}: {np.squeeze(cost)}")
        if i % 100 == 0:
            costs.append(cost)
    return parameters, costs

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print(f"Cost after iteration {i}: {np.squeeze(cost)}")
        if i % 100 == 0:
            costs.append(cost)
    return parameters, costs

def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    m_train, num_px, m_test = train_x_orig.shape[0], train_x_orig.shape[1], test_x_orig.shape[0]
    train_x = train_x_orig.reshape(m_train, -1).T / 255.
    test_x = test_x_orig.reshape(m_test, -1).T / 255.

    n_x = train_x.shape[0]
    n_h = 7
    n_y = 1
    parameters2, costs2 = two_layer_model(train_x, train_y, (n_x, n_h, n_y), num_iterations=2500, print_cost=True)
    plot_costs(costs2)
    _ = predict(train_x, train_y, parameters2)
    _ = predict(test_x, test_y, parameters2)

    layers_dims = [12288, 20, 7, 5, 1]
    parametersL, costsL = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)
    _ = predict(train_x, train_y, parametersL)
    _ = predict(test_x, test_y, parametersL)
    print_mislabeled_images(classes, test_x, test_y, predict(test_x, test_y, parametersL))

if __name__ == "__main__":
    main()
