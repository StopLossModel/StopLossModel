import tensorflow as tf
from tensorflow.keras import backend as K

######################### Util Functions #######################
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def mean_directional_accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, y_pred.shape[1]))
    result = (K.sign(y_true[1:] - y_true[:-1]) == K.sign(y_pred[1:] - y_pred[:-1]))
    return K.mean(result)

def custom_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, y_pred.shape[1]))
    # next day prices
    yt_next = y_true[1:]
    yp_next = y_pred[1:]

    # today prices
    yt_tdy = y_true[:-1]
    yp_tdy = y_pred[:-1]

    # subtract and get diff
    yt_diff = tf.subtract(yt_next, yt_tdy)
    yp_diff = tf.subtract(yp_next, yp_tdy)

    standard = tf.zeros_like(yp_diff)

    # compare with standard; if true: Up, else Down
    yt_move = tf.greater_equal(yt_diff, standard)
    yp_move = tf.greater_equal(yp_diff, standard)

    # indices where directions are not same
    condition = tf.not_equal(yt_move, yp_move)
    condition = tf.reshape(condition, [-1])

    # print(condition.shape)
    indices = tf.where(condition)

    ones = tf.ones_like(indices)
    indices = tf.add(indices, ones)

    # directional loss
    ones_yp = tf.ones_like(y_pred)
    dir_loss = tf.Variable(lambda: ones_yp, dtype='float32')
    updates = K.cast(tf.ones_like(indices), dtype='float32')

    # penalty
    alpha = 2000

    # print("loss shapes")
    # print(ones_yp.shape)
    # print(dir_loss.shape)
    # print(indices.shape)
    # print(updates.shape)

    dir_loss = tf.compat.v1.scatter_nd_update(dir_loss, indices, alpha*updates)

    c_loss = K.mean(tf.multiply(K.square(y_true - y_pred), dir_loss), axis=-1)

    return c_loss
