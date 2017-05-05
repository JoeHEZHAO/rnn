'''
Author: Manish Sapkota
Data: 05/02/2017
Description: optimized RNN implementation with tensorflow
Each x is a sentence and y is the shifted version of the axis
x starts with start_token and y ends with end_token
https://nbviewer.jupyter.org/github/rdipietro/tensorflow-notebooks/blob/master/tensorflow_scan_examples/tensorflow_scan_examples.ipynb#training
'''
import numpy as np
import tensorflow as tf
from utils import softmax

class TenRNN(object):
    '''
    This function will initialize our simple RNN and it learning parameters
    '''

    def __init__(self, word_dim, hidden_dim=100, bptt_steps=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_steps = bptt_steps

        # Random initialization of the network parameters
        # Here wer are creating just three layered RNN
        # np.random.uniform(low,high,tuplesize)
        # U - mapping from input to hidden
        # W - mapping from hidden to hidden across time
        # V - mapping from hidden to output
        # 1/sqrt(n) where n is number of previous layers
        U = np.random.uniform(-np.sqrt(1. / word_dim),
                              np.sqrt(1. / word_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / hidden_dim),
                              np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1. / hidden_dim),
                              np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))

        self.U = tf.Variable(U, name="U", dtype=tf.float32)
        self.W = tf.Variable(W, name="W", dtype=tf.float32)
        self.V = tf.Variable(V, name="V", dtype=tf.float32)

        self._inputs = tf.placeholder(tf.float32,
                                      shape=[None, word_dim],
                                      name='inputs')
        self._targets = tf.placeholder(tf.float32,
                                       shape=[None, ],
                                       name='targets')

        with tf.variable_scope('model'):
            self._states, self._logits, self._predictions = self._compute_predictions()
            self._loss = self._compute_loss()

    def _vanilla_rnn_step(self, st_prev, xt):
        return tf.matmul(st_prev, self.W) #+ tf.matmul(xt, self.U)

    def _compute_predictions(self):
        with tf.variable_scope('states'):
            init_states = tf.zeros([self.hidden_dim],
                                   name='initial_state')
                        
            states = tf.scan(self._vanilla_rnn_step,
                             self.inputs,
                             initializer=init_states,
                             name='states')

        with tf.variable_scope('predictions'):
            logits = tf.matmul(states, self.V)
            predictions = tf.nn.softmax(logits)

        return states, logits, predictions

    def _compute_loss(self):
        ''' Compute cross entropy loss with softmax '''
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmaxx_cross_entropy_with_logits(labels=self.argets,
                                                                                  logits=self.logits),
                                  name='loss')
            return loss
    
    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def states(self):
        return self._states
    
    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions

    @property
    def loss(self):
        return self._loss
