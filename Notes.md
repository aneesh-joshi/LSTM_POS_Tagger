# Notes on LSTM POS Tagger Shapes

```
X : numpy array of shape (No. of sample, Padding Length) 
						  Example : 64, 1000

						  [ [0, 0, ...., 52, 16, 23],
						    [0, 0, ...., 23, 64, 12]]
						   ^ this has shape (2, 1000)  since padding length is 1000
						   it corresponds to sentences
						   [ [pad, pad, ...., I, am, happy],
						     [pad, pad, ...., I, am, sad]]
```

```
y : numpy array of shape (No. of samples, No. of tags/emotions)
						  Example : 64, 6   (since there are 6 emotions)

						  [ [0, 0, 1, 0, 0, 0],
						  	[0, 1, 0, 0, 0, 0]]
						  	^ this has shape (2, 6) {see the X above for comparison}
						  	it corresponds to emotions
						  	[['happy'],
						  	 ['sad']]

						  BUT in your case, we aren't doing classification
						  so: instead of y being one hot, it will be something like
						  [ [0.2, 0.1, 0.6, 0, 0, 0.1],   # 60% happy, 10% sad, etc
						    [0.1, 0.7, 0, 0, 0.2, 0]]     # 70% sad, etc
```

**SO, no one hots required at all**

## Why no one hots?
Because, keras already handles the one hotting for you through the Embedding Layer

```
34 -> EmbeddingLayer(size = n_unique_vocabulary_words + 2 ) -> [0,..., 1, ... 0] shape:(1, n_unique_vocab + 2 )
```

So, essentially, you pass in 
```
[0, 0, 0, ..., 52, 16, 23] -> EmbeddingLayer -> [[1, 0, ...],
												 [1, 0, ...],
												 .
												 .
												 .
												 [0,..,1,.0],  #52 
												 [0,..,1,.0],  #16
												 [0,..,1,.0]]  #23
```

## Why n_unique_vocab + 2?
1. Because we have a padding of 0 which shouldn't correspond to any word
2. We want to make use of the <UNK> token
3. the remaining are for your words

## Now there are 2 scenarios:
### 1. Using w2v embeddings
In this, you will just take the pretrained word embeddings as a numpy array and insert it into
the keras Embedding Layer, taking care that
i. they start from index 1
ii. index 0 should be kept random for the 0 pad
iii. The last index should be kept random for the UNK token

Note: UNK token will not be encountered during training, only during testing
      UNK token should have the largest integer index as it is the last word in vocab

### 2. Not using w2v embeddings
Use the default embedding layer, it will randomly initialize weights and train

**Note:** I haven't taken care of pads + UNK in my implementation.