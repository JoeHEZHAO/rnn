import tensorflow as tf


class Optimizer(object):

    def __init__(self,
                 loss,
                 initial_learning_rate,
                 num_steps_per_decay,
                 decay_rate,
                 max_global_norm=1.0):

        trainables = tf.trainable_variables()
        grads = tf.gradients(loss, trainables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
        grad_var_pairs = zip(grads, trainables)

        global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate,
            global_step,
            num_steps_per_decay,
            decay_rate,
            staircase=True)
        optimizer = tf.train.AdagradOptimizer(learning_rate)

        self._optimize_op = optimizer.apply_gradients(grad_var_pairs,
                                                      global_step=global_step)


    @property
    def optimize_op(self):
        return self._optimize_op