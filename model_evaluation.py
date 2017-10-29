from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

with open('data.pkl', 'rb') as f:
	X_train, Y_train, word2int, int2word, tag2int, int2tag = pickle.load(f)

	del X_train
	del Y_train

# statement = ['he', 'is', 'running']
# statement = 'john is expected to race tomorrow'.split()
# np bez vbn in nn nn

# statement = 'send me some photos of your body'.split()
# vb
# ppo
# dti
# nns
# in
# pp$
# nn

statement = 'i want to dance with a girl'.split()
# ppss
# vb
# in
# nn
# in
# at
# nn

tokenised_statement = []

for word in statement:
	tokenised_statement.append(word2int[word])

tokenised_statement = np.asarray([tokenised_statement])

print(tokenised_statement)


tokenised_statement = pad_sequences(tokenised_statement, maxlen=100)

print(tokenised_statement)

model = load_model('initial_model.h5')

prediction = model.predict(tokenised_statement)

print(prediction.shape)

for pred in prediction[0]:
	# print(pred, "LOOL")
	try:
		print(int2tag[np.argmax(pred)])
	except:
		pass
		# print('NA')
