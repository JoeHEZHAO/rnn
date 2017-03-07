import numpy as np
import simpleRNN as rnn

np.random.seed(10)
vocabulary_size = 8000
X_train = np.random.uniform(0,1,(10, vocabulary_size))
model = rnn.SRNN(vocabulary_size)
o, s = model.fp(X_train[9])
print o.shape
print o