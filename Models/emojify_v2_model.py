import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, Activation, Embedding
)

# --------------------------------------------------
# Pretrained Embedding Layer
# --------------------------------------------------

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Create a Keras Embedding layer loaded with pretrained GloVe vectors.
    """
    vocab_size = len(word_to_index) + 1
    any_word = next(iter(word_to_vec_map.keys()))
    emb_dim = word_to_vec_map[any_word].shape[0]

    emb_matrix = np.zeros((vocab_size, emb_dim))

    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=emb_dim,
        trainable=False
    )

    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


# --------------------------------------------------
# Emojify V2 Model
# --------------------------------------------------

def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    Creates the Emojify-V2 model graph.
    """

    # Input layer
    sentence_indices = Input(shape=input_shape, dtype='int32')

    # Embedding layer
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)

    # First LSTM layer
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)

    # Second LSTM layer
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)

    # Dense + Softmax
    X = Dense(5)(X)
    X = Activation('softmax')(X)

    # Create model
    model = Model(inputs=sentence_indices, outputs=X)

    return model
