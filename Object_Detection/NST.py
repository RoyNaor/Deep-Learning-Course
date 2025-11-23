import tensorflow as tf
import numpy as np


#########################################################
# 1. CONTENT COST
#########################################################

def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost using activations of C and G.
    """
    a_C = content_output[-1]
    a_G = generated_output[-1]

    # Retrieve dimensions
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Unroll
    a_C_unrolled = tf.reshape(a_C, shape=[n_H * n_W * n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[n_H * n_W * n_C])

    # Content cost
    J_content = tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled)) / (4 * n_H * n_W * n_C)

    return J_content


#########################################################
# 2. GRAM MATRIX
#########################################################

def gram_matrix(A):
    """
    Computes Gram Matrix: GA = A * A^T
    """
    GA = tf.matmul(A, A, transpose_b=True)
    return GA


#########################################################
# 3. STYLE COST FOR A SINGLE LAYER
#########################################################

def compute_layer_style_cost(a_S, a_G):
    """
    Computes style cost for a single layer.
    """
    # Retrieve dimensions
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape to (n_C, n_H*n_W)
    a_S = tf.reshape(a_S, shape=[n_H * n_W, n_C])
    a_S = tf.transpose(a_S)

    a_G = tf.reshape(a_G, shape=[n_H * n_W, n_C])
    a_G = tf.transpose(a_G)

    # Gram matrices
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Layer style cost
    J_style_layer = tf.reduce_sum(tf.square(GS - GG)) / (4 * (n_C ** 2) * ((n_H * n_W) ** 2))

    return J_style_layer


#########################################################
# 4. TOTAL COST
#########################################################

@tf.function()
def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes total style-transfer cost.
    """
    J = alpha * J_content + beta * J_style
    return J


#########################################################
# 5. TRAIN STEP (Gradient Descent)
#########################################################

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

@tf.function()
def train_step(generated_image):
    """
    Performs one step of gradient descent on generated_image.
    Note: Uses globally defined vgg_model_outputs, a_S, a_C.
    """
    with tf.GradientTape() as tape:
        
        # 1 line: compute activations for generated image
        a_G = vgg_model_outputs(generated_image)

        # 1 line: compute style cost
        J_style = compute_style_cost(a_S, a_G)

        # 1 line: compute content cost
        J_content = compute_content_cost(a_C, a_G)

        # 1 line: total cost
        J = total_cost(J_content, J_style)

    grad = tape.gradient(J, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))

    return J
