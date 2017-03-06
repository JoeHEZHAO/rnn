'''
Author: Manish Sapkota
Data: 03/06/2017
Description: simple RNN implementation 
Followed the tutorial from original source 
http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
'''
import numpy as np

class SRNN:
    '''
    This function will initialize our simple RNN and it learning parameters
    '''
    def __init__(self,word_dim,hidden_dim=100, bptt_steps=4):
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
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim,word_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim,hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim,hidden_dim))
    
    '''
    Forward propagation
    '''
    def fp(self,x):
        # total number of time steps
        T = len(x)

        # save all the previous states since we need it for the computation
        s = np.zeros((T+1,self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)

        # The outputs at each time step. Save them for later as well
        o = np.zeros((T, self.word_dim))

        for t in np.arange(T):
            s[t] = np.tanh(self.U[:,x[t]] +self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        
        return [o,s]
    
    SRNN.fp = fp

    '''
    Prediction
    '''
    def predict(self,x):
        o, s = self.fp(x)
        return np.argmax(o, axis=1) # returns the index of maximum apply accross the column

    SRNN.predict = predict

    def calculate_sum_loss(self, x, y):
        L = 0
        # loop through each of the sentences
        for i in np.arange(len(y)):
            o, s = self.fp(x[i])
            correct_word_predictions =  o[np.arange(len(y[i])), y[i]]
            # Add the loss
            L+=-1 * np.sum(np.log(correct_word_predictions))

        return L
    
    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_sum_loss(x,y)/N
    
    SRNN.calculate_sum_loss = calculate_sum_loss
    SRNN.calculate_loss = calculate_loss

    

