import pickle
import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

with open('PickledData/data.pkl', 'rb') as f:
	X_train, Y_train, word2int, int2word, tag2int, int2tag = pickle.load(f)

	del X_train
	del Y_train

model = load_model('Models/model.h5')

# Example Sentences=========================
# john is   expected  to  race  tomorrow
# np   bez  vbn       in  nn    nn

#  send me  some photos of  that tree
#  vb   ppo dti  nns    in  pp   nn

#  i     want  to  dance with a   girl
#  ppss  vb    in  nn    in   at  nn


sentences = ["he is running".split(),
			 "aneesh is expected to race tomorrow".split(),
			 "i want to dance with a girl".split(),
			 "i am Aneesh".split()]

for sentence in sentences:

	tokenized_sentence = []

	for word in sentence:
		if word in word2int:
			tokenized_sentence.append(word2int[word])
		else:
			tokenized_sentence.append(word2int['<UNK>'])

	tokenized_sentence = np.asarray([tokenized_sentence])
	padded_tokenized_sentence = pad_sequences(tokenized_sentence, maxlen=100)

	print('\n\n')
	print('The sentence is ', sentence)
	print('The tokenized sentence is ',tokenized_sentence)
	print('The padded tokenized sentence is ', padded_tokenized_sentence)

	prediction = model.predict(padded_tokenized_sentence)

	for i, pred in enumerate(prediction[0]):
		print(int2word[padded_tokenized_sentence[0][i]], int2tag[np.argmax(pred)])
		# try:
		# print(padded_tokenized_sentence[i] , "  ::  " , int2tag[np.argmax(pred)])
		# except:
		# 	pass
		# 	print('NA')