import numpy as np

# --------------------------------------------------
# Cosine Similarity
# --------------------------------------------------

def cosine_similarity(u, v):
    """
    Compute cosine similarity between two vectors.
    """
    if np.all(u == v):
        return 1

    dot = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if np.isclose(norm_u * norm_v, 0, atol=1e-32):
        return 0

    return dot / (norm_u * norm_v)


# --------------------------------------------------
# Word Analogy
# --------------------------------------------------

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    a is to b as c is to ?
    """
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    e_a = word_to_vec_map[word_a]
    e_b = word_to_vec_map[word_b]
    e_c = word_to_vec_map[word_c]

    max_cosine_sim = -100
    best_word = None

    for w in word_to_vec_map.keys():
        if w == word_c:
            continue

        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word


# --------------------------------------------------
# Neutralize (Debias neutral words)
# --------------------------------------------------

def neutralize(word, bias_axis, word_to_vec_map):
    """
    Remove bias of a neutral word by projecting it
    onto the space orthogonal to the bias axis.
    """
    e = word_to_vec_map[word]

    e_biascomponent = np.dot(e, bias_axis) / np.dot(bias_axis, bias_axis) * bias_axis
    e_debiased = e - e_biascomponent

    return e_debiased


# --------------------------------------------------
# Equalize (Debias gendered word pairs)
# --------------------------------------------------

def equalize(pair, bias_axis, word_to_vec_map):
    """
    Equalize a pair of gender-specific words.
    """
    w1, w2 = pair
    e_w1 = word_to_vec_map[w1]
    e_w2 = word_to_vec_map[w2]

    # Mean vector
    mu = (e_w1 + e_w2) / 2

    # Decompose mean into bias and orthogonal components
    mu_B = np.dot(mu, bias_axis) / np.dot(bias_axis, bias_axis) * bias_axis
    mu_orth = mu - mu_B

    # Bias components
    e_w1B = np.dot(e_w1, bias_axis) / np.dot(bias_axis, bias_axis) * bias_axis
    e_w2B = np.dot(e_w2, bias_axis) / np.dot(bias_axis, bias_axis) * bias_axis

    # Correct bias components
    corrected_e_w1B = (
        np.sqrt(abs(1 - np.linalg.norm(mu_orth) ** 2))
        * (e_w1B - mu_B)
        / np.linalg.norm(e_w1 - mu_orth)
    )

    corrected_e_w2B = (
        np.sqrt(abs(1 - np.linalg.norm(mu_orth) ** 2))
        * (e_w2B - mu_B)
        / np.linalg.norm(e_w2 - mu_orth)
    )

    # Final equalized vectors
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    return e1, e2
