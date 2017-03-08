import numpy as np

def softmax(x):
    xs = np.exp(x - np.max(x)) # center values to avoid overflow
    return xs/np.sum(xs)

def save_model_parameters_theano(outfile, model):
    U, V, W = model.U, model.V, model.W
    np.savez(outfile, U=U, V=V, W=W)
    print("Saved model parameters to %s." % outfile)

def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1]))

def save_train_data(outfile, X_train, y_train):
    np.savez(outfile, X_train=X_train, y_train=y_train)
    print("Saved train data to %s" % outfile)

def load_train_data(path):
    npzfile =  np.load(path)
    X_train, y_train = npzfile["X_train"], npzfile["y_train"]
    print("Loaded train data from %s" % path)
    return X_train, y_train