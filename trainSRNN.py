#! /usr/bin/env python
'''
File to create the dummy training data and test all that we have learned for the simpleRNN
Use same to modify later for tensorflow
'''
import csv
import itertools
import nltk
import numpy as np

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"  # for all the words that are not in the vocabulary
start_token = "SENTENCE_START"
end_token = "SENTENCE_END"

print("Reading the redit comment csv file")

with open('data/redit_comment.csv', 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.__next__()

    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])

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
      (vocab[-1][0], vocab[-1][1]))

for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [
        w if w in word_to_index else unknown_token for w in sent]

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]]
                      for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[:-1]]
                      for sent in tokenized_sentences])
