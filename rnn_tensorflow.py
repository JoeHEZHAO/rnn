'''
Author: Manish Sapkota
Data: 05/02/2017
Description: optimized RNN implementation with tensorflow
Each x is a sentence and y is the shifted version of the axis
x starts with start_token and y ends with end_token
'''
import numpy as np
import tensorflow as tf
from utils import softmax

#TODO::Description of all the input and out expected should
#be defined for the functions

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
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))

        self.U = tf.Variable(U, name="U", dtype=tf.float32)
        self.W = tf.Variable(W, name="W", dtype=tf.float32)
        self.V = tf.Variable(V, name="V", dtype=tf.float32)
        
    def __tensorflow_build__(self):
        U, V, W = self.U, self.V, self.W
        
        # inputs
        xs_ = tf.placeholder(shape=[None, None], 
                             dtype=tf.int32)
        ys_ = tf.placeholder(shape=[None], 
                             dtype=tf.int32)

        initial_state = tf.placeholder(shape=[None, self.hidden_dim], 
                                       dtype=tf.float32, 
                                       name='initial_state')
        
        def step(st_prev, xt):
            return tf.matmul(st_prev, W) + 
                   tf.matmul(x, U)
        


    



