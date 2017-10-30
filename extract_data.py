import pickle
import numpy as np
import os

files = os.listdir('brown/')

# In case your system can't handle all 500 samples
# set the number of samples to a reasonable number like 20
n_sample_files = 20

print('TOTAL NO. OF FILES ', len(files), '\n')
print('RUNNING ON ', n_sample_files, ' FILES\n')

raw_corpus = ''

for file in files[0:n_sample_files]:
	with open('brown/' + file) as f:
		raw_corpus = raw_corpus + '\n' + f.read()

corpus = raw_corpus.split('\n')

print('CORPUS SIZE', len(corpus), '\n')

X_train = []
Y_train = []

words = []
tags = []

with_slash = False
n_omitted = 0

for line in corpus:
	if(len(line)>0):
		tempX = []
		tempY = []
		for word in line.split():
			try:			
				w, tag = word.split('/')
			except:
				# with_slash = True
				n_omitted = n_omitted + 1
				break

			w = w.lower()
			words.append(w)
			tags.append(tag)

			tempX.append(w)
			tempY.append(tag)
		
		X_train.append(tempX)
		Y_train.append(tempY)


print('OMITTED sentences: ', n_omitted, '\n')
print('TOTAL NO OF SAMPLES: ', len(X_train), '\n')


print('sample X_train: ', X_train[42], '\n')
print('sample Y_train: ', Y_train[42], '\n')

words = set(words)
tags = set(tags)

print('VOCAB SIZE: ', len(words))
print('TOTAL TAGS: ', len(tags))

assert len(X_train) == len(Y_train)


word2int = {}
int2word = {}

for i, word in enumerate(words):
	word2int[word] = i+1
	int2word[i+1] = word

tag2int = {}
int2tag = {}

for i, tag in enumerate(tags):
	tag2int[tag] = i+1
	int2tag[i+1] = tag

X_train_numberised = []
Y_train_numberised = []

for sentence in X_train:
	tempX = []
	for word in sentence:
		tempX.append(word2int[word])
	X_train_numberised.append(tempX)

for tags in Y_train:
	tempY = []
	for tag in tags:
		tempY.append(tag2int[tag])
	Y_train_numberised.append(tempY)

print('sample X_train_numberised: ', X_train_numberised[42], '\n')
print('sample Y_train_numberised: ', Y_train_numberised[42], '\n')

X_train_numberised = np.asarray(X_train_numberised)
Y_train_numberised = np.asarray(Y_train_numberised)

pickle_files = [X_train_numberised, Y_train_numberised, word2int, int2word, tag2int, int2tag]

# with open('data.pkl', 'wb') as f:
	# pickle.dump(pickle_files, f)

print('Saved as pickle file')
