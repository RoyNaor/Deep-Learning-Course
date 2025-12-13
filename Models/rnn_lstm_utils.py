import numpy as np

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------------------------------------------------
# RNN CELL FORWARD
# --------------------------------------------------

def rnn_cell_forward(xt, a_prev, parameters):
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache


# --------------------------------------------------
# RNN FORWARD
# --------------------------------------------------

def rnn_forward(x, a0, parameters):
    caches = []

    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    a_next = a0

    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)

    return a, y_pred, (caches, x)


# --------------------------------------------------
# RNN CELL BACKWARD
# --------------------------------------------------

def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, xt, parameters) = cache
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]

    dtanh = (1 - a_next ** 2) * da_next

    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    dba = np.sum(dtanh, axis=1, keepdims=True)

    return {
        "dxt": dxt,
        "da_prev": da_prev,
        "dWax": dWax,
        "dWaa": dWaa,
        "dba": dba
    }


# --------------------------------------------------
# RNN BACKWARD (BPTT)
# --------------------------------------------------

def rnn_backward(da, caches):
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]

    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da_prevt = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
        dx[:, :, t] = gradients["dxt"]
        da_prevt = gradients["da_prev"]
        dWax += gradients["dWax"]
        dWaa += gradients["dWaa"]
        dba += gradients["dba"]

    return {
        "dx": dx,
        "da0": da_prevt,
        "dWax": dWax,
        "dWaa": dWaa,
        "dba": dba
    }


# --------------------------------------------------
# LSTM CELL FORWARD
# --------------------------------------------------

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    Wf, bf = parameters["Wf"], parameters["bf"]
    Wi, bi = parameters["Wi"], parameters["bi"]
    Wc, bc = parameters["Wc"], parameters["bc"]
    Wo, bo = parameters["Wo"], parameters["bo"]
    Wy, by = parameters["Wy"], parameters["by"]

    concat = np.concatenate((a_prev, xt), axis=0)

    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)

    c_next = ft * c_prev + it * cct
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)

    yt_pred = softmax(np.dot(Wy, a_next) + by)

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    return a_next, c_next, yt_pred, cache


# --------------------------------------------------
# LSTM FORWARD
# --------------------------------------------------

def lstm_forward(x, a0, parameters):
    caches = []

    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape

    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    a_next = a0
    c_next = np.zeros((n_a, m))

    for t in range(T_x):
        a_next, c_next, yt, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = yt
        caches.append(cache)

    return a, y, c, (caches, x)


# --------------------------------------------------
# LSTM CELL BACKWARD
# --------------------------------------------------

def lstm_cell_backward(da_next, dc_next, cache):
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    n_x, m = xt.shape
    n_a, m = a_next.shape

    concat = np.concatenate((a_prev, xt), axis=0)

    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dc = dc_next + da_next * ot * (1 - np.tanh(c_next) ** 2)

    dcct = dc * it * (1 - cct ** 2)
    dit = dc * cct * it * (1 - it)
    dft = dc * c_prev * ft * (1 - ft)

    dWf = np.dot(dft, concat.T)
    dWi = np.dot(dit, concat.T)
    dWc = np.dot(dcct, concat.T)
    dWo = np.dot(dot, concat.T)

    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)

    da_prev = (
        np.dot(parameters["Wf"][:, :n_a].T, dft) +
        np.dot(parameters["Wi"][:, :n_a].T, dit) +
        np.dot(parameters["Wc"][:, :n_a].T, dcct) +
        np.dot(parameters["Wo"][:, :n_a].T, dot)
    )

    dc_prev = dc * ft

    dxt = (
        np.dot(parameters["Wf"][:, n_a:].T, dft) +
        np.dot(parameters["Wi"][:, n_a:].T, dit) +
        np.dot(parameters["Wc"][:, n_a:].T, dcct) +
        np.dot(parameters["Wo"][:, n_a:].T, dot)
    )

    return {
        "dxt": dxt,
        "da_prev": da_prev,
        "dc_prev": dc_prev,
        "dWf": dWf, "dbf": dbf,
        "dWi": dWi, "dbi": dbi,
        "dWc": dWc, "dbc": dbc,
        "dWo": dWo, "dbo": dbo
    }


# --------------------------------------------------
# LSTM BACKWARD (BPTT)
# --------------------------------------------------

def lstm_backward(da, caches):
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    dx = np.zeros((n_x, m, T_x))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))

    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))

    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))

    for t in reversed(range(T_x)):
        grads = lstm_cell_backward(
            da[:, :, t] + da_prevt,
            dc_prevt,
            caches[t]
        )

        dx[:, :, t] = grads["dxt"]
        da_prevt = grads["da_prev"]
        dc_prevt = grads["dc_prev"]

        dWf += grads["dWf"]
        dWi += grads["dWi"]
        dWc += grads["dWc"]
        dWo += grads["dWo"]

        dbf += grads["dbf"]
        dbi += grads["dbi"]
        dbc += grads["dbc"]
        dbo += grads["dbo"]

    return {
        "dx": dx,
        "da0": da_prevt,
        "dWf": dWf, "dbf": dbf,
        "dWi": dWi, "dbi": dbi,
        "dWc": dWc, "dbc": dbc,
        "dWo": dWo, "dbo": dbo
    }
