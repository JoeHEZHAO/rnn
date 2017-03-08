#! /usr/bin/env python
'''
File to create the dummy training data and test all that we have learned for the simpleRNN
Use same to modify later for tensorflow
'''
import csv
import itertools
import nltk
import numpy as np
import simpleRNN as srnn
import os
import utils

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

#nltk.download('book')

def train_rnn_sgd(model, X_train, y_train, learning_rate=0.05, nepoch=5, evaluate_loss_after=5):
    losses = []
    num_examples_seen = 0
    for epoch in np.arange(nepoch):
        if(epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time =  datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()

            # ADDED! Saving model oarameters
            save_model_parameters_theano("./data/srnn-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)

        for i in range(len(y_train)):
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"  # for all the words that are not in the vocabulary
start_token = "SENTENCE_START"
end_token = "SENTENCE_END"

print("Reading the redit comment csv file")

#with open('data/redit_comment.csv', 'r') as f:
with open('data/redit_comment.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    try:
        reader.next()
    except:
        reader.__next__()

    # use this for python 3.0
    #sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    

    sentences = ["%s %s %s" % (start_token, x, end_token) for x in sentences]

print("Parsed %d sentences." % (len(sentences)))

tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." %
      (vocab[-1][0], vocab[-1][1])) #last element in vocab

for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [
        w if w in word_to_index else unknown_token for w in sent]

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] 
                      for sent in tokenized_sentences])# sent[:-1] all other than last because last is end token
y_train = np.asarray([[word_to_index[w] for w in sent[1:]]
                      for sent in tokenized_sentences])# sent[1:] remove start token for the label

model = srnn(vocabulary_size, hidden_dim = 100)
t1 = time.time()
model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
t2 = time.time()

print("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))

if _MODEL_FILE != None:
    load_model_parameters_theano(_MODEL_FILE, model)

train_rnn_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)