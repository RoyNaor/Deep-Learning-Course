# Trigger Word Detection – FULL MODEL & DATA PIPELINE

import numpy as np
from pydub import AudioSegment
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, Dropout,
    GRU, TimeDistributed, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --------------------------------------------------
# Helper functions (provided in assignment context)
# --------------------------------------------------
# Assumes the following exist in your environment:
# - get_random_time_segment
# - match_target_amplitude
# - graph_spectrogram

# --------------------------------------------------
# UNQ_C1 – Check overlap
# --------------------------------------------------

def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time
    overlap = False
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
            break
    return overlap

# --------------------------------------------------
# UNQ_C2 – Insert audio clip
# --------------------------------------------------

def insert_audio_clip(background, audio_clip, previous_segments):
    segment_ms = len(audio_clip)

    segment_time = get_random_time_segment(segment_ms)
    retry = 5
    while is_overlapping(segment_time, previous_segments) and retry >= 0:
        segment_time = get_random_time_segment(segment_ms)
        retry -= 1

    if not is_overlapping(segment_time, previous_segments):
        previous_segments.append(segment_time)
        new_background = background.overlay(audio_clip, position=segment_time[0])
    else:
        new_background = background
        segment_time = (10000, 10000)

    return new_background, segment_time

# --------------------------------------------------
# UNQ_C3 – Insert ones in label vector
# --------------------------------------------------

def insert_ones(y, segment_end_ms):
    _, Ty = y.shape
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    if segment_end_y < Ty:
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < Ty:
                y[0, i] = 1
    return y

# --------------------------------------------------
# UNQ_C4 – Create training example
# --------------------------------------------------

def create_training_example(background, activates, negatives, Ty):
    background = background - 20

    y = np.zeros((1, Ty))
    previous_segments = []

    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    for activate in random_activates:
        background, segment_time = insert_audio_clip(background, activate, previous_segments)
        _, segment_end = segment_time
        y = insert_ones(y, segment_end)

    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    for negative in random_negatives:
        background, _ = insert_audio_clip(background, negative, previous_segments)

    background = match_target_amplitude(background, -20.0)
    background.export("train.wav", format="wav")

    x = graph_spectrogram("train.wav")
    return x, y

# --------------------------------------------------
# UNQ_C5 – Model definition
# --------------------------------------------------

def modelf(input_shape):
    X_input = Input(shape=input_shape)

    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)

    X = GRU(128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)

    X = GRU(128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)

    X = TimeDistributed(Dense(1, activation='sigmoid'))(X)

    model = Model(inputs=X_input, outputs=X)
    return model

# --------------------------------------------------
# Compile helper
# --------------------------------------------------

def compile_model(model):
    opt = Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
