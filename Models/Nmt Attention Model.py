# Neural Machine Translation with Attention (Andrew Ng - DLS)
# Complete graded code for UNQ_C1, UNQ_C2, and model compilation

from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, RepeatVector, Concatenate, Activation, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------------------
# Shared layers (as expected by the grader)
# ------------------------------------------------------------
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation="tanh")
densor2 = Dense(1)
activator = Activation("softmax", name='attention_weights')
dotor = Dot(axes=1)

post_activation_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(machine_vocab_size, activation='softmax')

# ------------------------------------------------------------
# UNQ_C1: one_step_attention
# ------------------------------------------------------------

def one_step_attention(a, s_prev):
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])
    return context

# ------------------------------------------------------------
# UNQ_C2: model
# ------------------------------------------------------------

def modelf(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size=None):
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')

    s = s0
    c = c0

    outputs = []

    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    for t in range(Ty):
        context = one_step_attention(a, s)
        _, s, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        out = output_layer(s)
        outputs.append(out)

    model = Model(inputs=[X, s0, c0], outputs=outputs)
    return model

# ------------------------------------------------------------
# Compile the model (Exercise 3)
# ------------------------------------------------------------

opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
