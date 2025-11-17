#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ResNet-50 implemented from scratch using TensorFlow/Keras
Includes:
    - identity_block()
    - convolutional_block()
    - ResNet50()
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Add, Dense, Activation, ZeroPadding2D, Flatten,
    Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import random_uniform, glorot_uniform


# ---------------------------------------------------------
# IDENTITY BLOCK
# ---------------------------------------------------------

def identity_block(X, f, filters, initializer=random_uniform):
    """
    Identity block where input and output dimensions match
    """
    F1, F2, F3 = filters

    X_shortcut = X

    # First
    X = Conv2D(F1, (1,1), strides=(1,1), padding='valid',
               kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Second
    X = Conv2D(F2, (f,f), strides=(1,1), padding='same',
               kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Third
    X = Conv2D(F3, (1,1), strides=(1,1), padding='valid',
               kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    # Add shortcut
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# ---------------------------------------------------------
# CONVOLUTIONAL BLOCK
# ---------------------------------------------------------

def convolutional_block(X, f, filters, s=2, initializer=glorot_uniform):
    """
    Convolutional block used when spatial dimensions change
    """
    F1, F2, F3 = filters

    X_shortcut = X

    # First
    X = Conv2D(F1, (1,1), strides=(s,s), padding='valid',
               kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Second
    X = Conv2D(F2, (f,f), strides=(1,1), padding='same',
               kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Third
    X = Conv2D(F3, (1,1), strides=(1,1), padding='valid',
               kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    # Shortcut Conv
    X_shortcut = Conv2D(F3, (1,1), strides=(s,s), padding='valid',
                        kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    # Add
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# ---------------------------------------------------------
# RESNET-50 ARCHITECTURE
# ---------------------------------------------------------

def ResNet50(input_shape=(64,64,3), classes=6):
    """
    Full ResNet-50 architecture
    """

    X_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(X_input)

    # Stage 1
    X = Conv2D(64, (7,7), strides=(2,2),
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), strides=(2,2))(X)

    # Stage 2
    X = convolutional_block(X, 3, [64,64,256], s=1)
    X = identity_block(X, 3, [64,64,256])
    X = identity_block(X, 3, [64,64,256])

    # Stage 3
    X = convolutional_block(X, 3, [128,128,512], s=2)
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])

    # Stage 4
    X = convolutional_block(X, 3, [256,256,1024], s=2)
    for _ in range(5):
        X = identity_block(X, 3, [256,256,1024])

    # Stage 5
    X = convolutional_block(X, 3, [512,512,2048], s=2)
    X = identity_block(X, 3, [512,512,2048])
    X = identity_block(X, 3, [512,512,2048])

    # Average Pooling
    X = AveragePooling2D((2,2))(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax',
              kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X)
    return model


# ---------------------------------------------------------
# MAIN (optional demo)
# ---------------------------------------------------------

if __name__ == "__main__":
    model = ResNet50(input_shape=(64,64,3), classes=6)
    model.summary()
