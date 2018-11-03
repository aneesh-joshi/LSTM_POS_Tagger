from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

TEST_DATA = "testData/test"
with open('PickledData/data.pkl', 'rb') as f:
	X_train, Y_train, word2int, int2word, tag2int, int2tag = pickle.load(f)

	del X_train
	del Y_train


def get_tokenized(sentence):
	tokenized_sentence = []
	for word in sentence:
		tokenized_sentence.append(word2int[word])
	tokenized_sentence = np.asarray([tokenized_sentence])
	padded_tokenized_sentence = pad_sequences(tokenized_sentence, maxlen=100)
	return tokenized_sentence, padded_tokenized_sentence


def print_predicted(prediction):
	for i, pred in enumerate(prediction[0]):
		if i >= len(list(enumerate(prediction[0]))) - len(sentence):
			try:
				print(int2word[padded_tokenized_sentence[0][i]], int2tag[np.argmax(pred)])
			except KeyError:
				pass


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

sentence = 'i want to dance with a dog'.split()
# ppss
# vb
# in
# nn
# in
# at
# nn

"""
	Predicting hardcoded sentences
"""

tokenized_sentence, padded_tokenized_sentence = get_tokenized(sentence)

print('The sentence is ', sentence)
print('The tokenized sentence is ',tokenized_sentence)
print('The padded tokenized sentence is ', padded_tokenized_sentence)

model = load_model('Models/model.h5')

prediction = model.predict(padded_tokenized_sentence)
print(prediction.shape)
print_predicted(prediction)

"""
	Predicting sentences from test file
	Sentences should be in form WORD/TAG WORD2/TAG2 ... (tags are there for future evaluation)
"""
print("Predicting from file...")
# Make two lists: correct tags and words to be used for later evaluation
test_sentences = []  # list of list of words per sentence
correct_tags = []  # list of list of tags per sentence

# Reading test data from file
with open(TEST_DATA) as test_f:
	test_corpus = test_f.readlines()
	for line in test_corpus:
		words = []
		tags = []
		if len(line) > 0:
			for word in line.split():
				try:
					w, tag = word.split('/')
					w = w.lower()
					words.append(w)
					tags.append(tag)
				except:
					pass
		test_sentences.append(words)
		correct_tags.append(tags)

for s in test_sentences:
	print('The sentence is ', sentence)
	tokenized_sentence, padded_tokenized_sentence = get_tokenized(s)
	prediction = model.predict(padded_tokenized_sentence)
	print_predicted(prediction)



