import numpy as np
import tensorflow as tf

###############################################
# Triplet Loss
###############################################

def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implements the triplet loss function.
    
    Arguments:
    y_true -- Required for Keras loss function signature (unused)
    y_pred -- List of tensors [anchor, positive, negative]
    
    Returns:
    loss -- Scalar loss value
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Distance between anchor and positive
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)

    # Distance between anchor and negative
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    # Basic loss
    basic_loss = pos_dist - neg_dist + alpha

    # Final loss
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss


###############################################
# Verify identity function
###############################################

def verify(image_path, identity, database, model):
    """
    Verify if the person in the image is the claimed identity.
    
    Returns:
    dist -- L2 distance between encodings
    door_open -- True/False if allowed
    """
    # Compute encoding
    encoding = img_to_encoding(image_path, model)

    # Distance to stored encoding
    dist = np.linalg.norm(encoding - database[identity])

    # Decision threshold: 0.7
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open


###############################################
# Face recognition (who is it?)
###############################################

def who_is_it(image_path, database, model):
    """
    Recognize the identity of the person in image_path.
    
    Returns:
    min_dist -- Minimum L2 distance
    identity -- Predicted name
    """
    # Compute encoding
    encoding = img_to_encoding(image_path, model)

    # Init
    min_dist = 100
    identity = None

    # Search database
    for name, db_enc in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name

    # Decision
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity
