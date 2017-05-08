#! /usr/bin/env python
'''
File to create the dummy training data and test all that we have learned for the simpleRNN
Use same to modify later for tensorflow
Original code from this guy:
https://nbviewer.jupyter.org/github/rdipietro/tensorflow-notebooks/blob/master/tensorflow_scan_examples/tensorflow_scan_examples.ipynb#training
'''
import csv
import itertools
#import nltk
import numpy as np
from simpleRNN import SRNN as srnn
import os
import time
import datetime
import sys
from utils import *
import shutil
from rnn_tensorflow import TenRNN
from optimizer import Optimizer
import tensorflow as tf


_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

# nltk.download('book')
def train(sess,
          model,
          optimizer,
          X_train,
          y_train,
          num_epoch,
          num_classes,
          logdir='./logdir'):

    if os.path.exists(logdir):
        shutil.rmtree(logdir)

    tf.summary.scalar('loss', model.loss)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss_ema = ema.apply([model.loss])
    loss_ema = ema.average(model.loss)
    tf.summary.scalar('loss_ema', loss_ema)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)

    sess.run(tf.global_variables_initializer())

    step = 0
    for epoch in np.arange(num_epoch):
        for i in range(len(y_train)):
            #x_one_hot = tf.one_hot(X_train[i], num_classes)
            #rnn_inputs = x_one_hot

            #rnn_inputs = tf.unstack(x_one_hot, axis=1)
            #x_one_hot = tf.one_hot(y_train[i], num_classes)
            #rnn_inputs = tf.unstack(x_one_hot, axis=1)
            x = np.array(X_train[i])
            y = np.array(y_train[i])

            x = x.reshape(-1, 1)
            y = x.reshape(-1, 1)
            loss_ema_, summary, _, _ = sess.run(
                [loss_ema, summary_op, optimizer.optimize_op, update_loss_ema],
                {model.inputs: x, model.targets: y})
            summary_writer.add_summary(summary, global_step=step)
            print('\rStep %d. Loss EMA: %.6f.' % (step + 1, loss_ema_))


vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"  # for all the words that are not in the vocabulary
start_token = "SENTENCE_START"
end_token = "SENTENCE_END"

print("Reading the redit comment csv file")

train_data_file = "./data/srnn-train-processed.npz"

if (not os.path.isfile(train_data_file)):
    # with open('data/redit_comment.csv', 'r') as f:
    with open('data/redit_comment.csv', 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        try:
            reader.next()
        except:
            reader.__next__()

        # use this for python 3.0
        #sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        sentences = itertools.chain(
            *[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])

        sentences = ["%s %s %s" % (start_token, x, end_token)
                     for x in sentences]

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
          (vocab[-1][0], vocab[-1][1]))  # last element in vocab

    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [
            w if w in word_to_index else unknown_token for w in sent]

    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]]
                          for sent in tokenized_sentences])  # sent[:-1] all other than last because last is end token
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]]
                          for sent in tokenized_sentences])  # sent[1:] remove start token for the label

    save_train_data(train_data_file, X_train, y_train)
else:
    X_train, y_train = load_train_data(train_data_file)

model = TenRNN(1, hidden_dim=100)
optimizer = Optimizer(model.loss,
                      initial_learning_rate=_LEARNING_RATE,
                      num_steps_per_decay=15000,
                      decay_rate=0.1,
                      max_global_norm=1.0)
sess = tf.Session()

t1 = time.time()
train(sess,
      model,
      optimizer,
      X_train,
      y_train,
      1,
      vocabulary_size,
      logdir='./logdir')
t2 = time.time()

print("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))

train(sess,
      model,
      optimizer,
      X_train,
      y_train,
      10,
      vocabulary_size,
      logdir='./logdir')
