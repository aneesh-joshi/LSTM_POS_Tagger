import numpy as np
import pickle

import sys
import os

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

with open('data.pkl', 'rb') as f:
  X_train, Y_train, word2int, int2word, tag2int, int2tag = pickle.load(f)

n_tags = len(tag2int)

X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
Y_train = pad_sequences(Y_train, maxlen=MAX_SEQUENCE_LENGTH)

Y_train = to_categorical(Y_train, num_classes= len(tag2int) + 1)


Y_train = Y_train.reshape((Y_train.shape[0]//MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH, len(tag2int) + 1))

print('TOTAL TAGS', len(tag2int))
print('TOTAL WORDS', len(word2int))


indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
nb_validation_samples = int(VALIDATION_SPLIT * X_train.shape[0])

x_train = X_train[:-nb_validation_samples]
y_train = Y_train[:-nb_validation_samples]

x_val = X_train[-nb_validation_samples*2:-nb_validation_samples]
y_val = Y_train[-nb_validation_samples*2:-nb_validation_samples]

x_test = X_train[-nb_validation_samples:]
y_test = Y_train[-nb_validation_samples:]

with open('Glove.pkl', 'rb') as f:
	embeddings_index = pickle.load(f)

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word2int) + 1, EMBEDDING_DIM))

for word, i in word2int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('Embedding matrix shape', embedding_matrix.shape)
print('x_train shape', x_train.shape)

embedding_layer = Embedding(len(word2int) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(69, return_sequences=True))(embedded_sequences)
preds = TimeDistributed(Dense(n_tags + 1, activation='softmax'))(l_lstm)
model = Model(sequence_input, preds)


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Bidirectional LSTM")
model.summary()

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=1, batch_size=50)

# predictions = model.predict(x_test)

# n_correct = 0
# n_total = 0

# for i, pred in enumerate(predictions[0]):
#   n_total = n_total + 1
#   try:
#     if np.argmax(pred) == np.argmax(y_test[i]):
#       n_correct = n_correct + 1
#   except:
#     pass
#     # print('NA')

print('TEST ACCURACY: ', model.evaluate(x_test, y_test, verbose=0))



model.save('initial_model.h5')
print('MODEL SAVED')
