#!/usr/bin/env python3
# coding: utf-8

"""
Alpaca / Not Alpaca Classifier
Transfer Learning + Fine Tuning using MobileNetV2
Dataset structure:
dataset/
    alpaca/
    not_alpaca/

Author: YOUR NAME
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
DATASET_DIR = "dataset/"

train_dataset = image_dataset_from_directory(
    DATASET_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset='training',
    seed=42
)

validation_dataset = image_dataset_from_directory(
    DATASET_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=42
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

class_names = train_dataset.class_names

# ---------------------------------------------------------
# 2. Data Augmentation
# ---------------------------------------------------------

def data_augmenter():
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip("horizontal"))
    data_augmentation.add(RandomRotation(0.2))
    return data_augmentation


data_augmentation = data_augmenter()
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# ---------------------------------------------------------
# 3. Transfer Learning Model (Exercise 2)
# ---------------------------------------------------------

def alpaca_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    input_shape = image_shape + (3,)

    base_model_path = (
        "imagenet_base_model/"
        "without_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=base_model_path
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    return model


model2 = alpaca_model()

# ---------------------------------------------------------
# 4. First Training Phase
# ---------------------------------------------------------

base_learning_rate = 0.001

model2.compile(
    optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

initial_epochs = 5
history = model2.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

# ---------------------------------------------------------
# 5. Fine Tuning (Exercise 3)
# ---------------------------------------------------------

base_model = model2.layers[4]  # MobileNetV2 inside the Functional model
base_model.trainable = True

fine_tune_at = 120

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.1 * base_learning_rate)
metrics = ['accuracy']

model2.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

fine_tune_epochs = 5
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model2.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset
)

# ---------------------------------------------------------
# 6. Plot Accuracy and Loss
# ---------------------------------------------------------

acc = [0.] + history.history['accuracy'] + history_fine.history['accuracy']
val_acc = [0.] + history.history['val_accuracy'] + history_fine.history['val_accuracy']

loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.show()

print("Training Finished!")
