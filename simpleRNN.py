'''
Author: Manish Sapkota
Data: 03/06/2017
Description: simple RNN implementation 
Followed the tutorial from original source 
http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
Each x is a sentence and y is the shifted version of the axis
x starts with start_token and y ends with end_token
'''
import numpy as np
from utils import softmax

#TODO::Description of all the input and out expected should be defined for the functions

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
        T = len(x) # length of the sentence is the time steps

        # save all the previous states since we need it for the computation
        s = np.zeros((T+1,self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)

        # The outputs at each time step. Save them for later as well
        o = np.zeros((T, self.word_dim))

        # for each time step. Each time step is a word
        for t in np.arange(T):
            s[t] = np.tanh(self.U[:,x[t]] +self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        
        return [o,s]
    
    #self.fp = fp

    '''
    Prediction
    '''
    def predict(self,x):
        o, s = self.fp(x)
        return np.argmax(o, axis=1) # returns the index of maximum apply accross the column

    #SRNN.predict = predict

    '''
    Total of the loss for the given samples
    I am assuming these will be batches
    '''
    def calculate_sum_loss(self, x, y):
        L = 0
        # loop through each of the sentences
        for i in np.arange(len(y)): # check the dimension for x and y
            o, s = self.fp(x[i])
            correct_word_predictions =  o[np.arange(len(y[i])), y[i]] # not sure what this line does
            # Add the loss
            L+=-1 * np.sum(np.log(correct_word_predictions))

        return L
    '''
    Compute the normalized loss
    '''
    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_sum_loss(x,y)/N
    
    #SRNN.calculate_sum_loss = calculate_sum_loss
    #SRNN.calculate_loss = calculate_loss

    '''
    Do the mathematics for the back-propagation should be straight forward
    '''
    def bptt(self, x, y):
        T = len(y)
        o, s = self.fp(x)

        # gradient of the parameters
        dLdU = np.zeros(self.U.shape)
        dLdW = np.zeros(self.W.shape)
        dLdV = np.zeros(self.V.shape)

        delta_o[np.arange(len(y)), y]  -= 1

        for t in np.arange(T) [::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            delta_t =  self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))

            for bptt_step in np.arange(max(0, t-self.bptt_steps), t+1)[::-1]: # similar to for i in range(100, -1, -1)
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                delta_t =  self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]
    
    #SRNN.bptt = bptt

    '''
    Dig into what is going on in the gradient_check
    This is important
    '''
    def gradient_check(self, x, y, h=0.001, error_threshold = 0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print ("+h Loss: %f" % gradplus)
                    print ("-h Loss: %f" % gradminus)
                    print ("Estimated_gradient: %f" % estimated_gradient)
                    print ("Backpropagation gradient: %f" % backprop_gradient)
                    print ("Relative Error: %f" % relative_error)
                    return
                it.iternext()

            print ("Gradient check for parameter %s passed." % (pname))
 
    #SRNN.gradient_check = gradient_check

    '''
    Single stochastic gradient descent
    '''
    def sgd_step(self, x, y, learning_rate=0.1):
        # get the gradients for the parameters
        dLdU, dLdV, dLDw = self.bptt(x, y)

        #update the parameters
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLDw



