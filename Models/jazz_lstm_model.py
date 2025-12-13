import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape, RepeatVector
from tensorflow.keras.utils import to_categorical


# --------------------------------------------------
# DJ MODEL (Training)
# --------------------------------------------------

def djmodel(Tx, LSTM_cell, densor, reshaper):
    n_values = densor.units
    n_a = LSTM_cell.units

    X = Input(shape=(Tx, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')

    a = a0
    c = c0
    outputs = []

    for t in range(Tx):
        x = X[:, t, :]
        x = reshaper(x)
        _, a, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)

    model = Model(inputs=[X, a0, c0], outputs=outputs)
    return model


# --------------------------------------------------
# INFERENCE MODEL (Generation)
# --------------------------------------------------

def music_inference_model(LSTM_cell, densor, Ty=100):
    n_values = densor.units
    n_a = LSTM_cell.units

    x0 = Input(shape=(1, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')

    a = a0
    c = c0
    x = x0
    outputs = []

    for t in range(Ty):
        _, a, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)

        x = tf.argmax(out, axis=-1)
        x = tf.one_hot(x, depth=n_values)
        x = RepeatVector(1)(x)

    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    return inference_model


# ------------------------
