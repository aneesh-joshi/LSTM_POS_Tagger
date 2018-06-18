from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from extract_data import get_data

# with open('PickledData/data.pkl', 'rb') as f:
X_train, Y_train, word2int, int2word, tag2int, int2tag = get_data()

del X_train
del Y_train

model = load_model('Models/model.h5')


# sentence = ['he', 'is', 'running']
# sentence = 'john is expected to race tomorrow'.split()
# np bez vbn in nn nn

# sentence = 'send me some photos of that tree'.split()
# vb
# ppo
# dti
# nns
# in
# pp$
# nn

sentence = 'i want to dance with a girl'.split()
# ppss
# vb
# in
# nn
# in
# at
# nn

tokenized_sentence = []

for word in sentence:
	tokenized_sentence.append(word2int[word])

tokenized_sentence = np.asarray([tokenized_sentence])
padded_tokenized_sentence = pad_sequences(tokenized_sentence, maxlen=100)

print('The sentence is ', sentence)
print('The tokenized sentence is ',tokenized_sentence)
print('The padded tokenized sentence is ', padded_tokenized_sentence)


prediction = model.predict(padded_tokenized_sentence)

print(sentence)
for pred in prediction[0]:
	try:
		print(int2tag[np.argmax(pred)])
	except:
		pass
		# print('NA')
