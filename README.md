# LSTM_POS_Tagger
A simple POS Tagger made with a Bidirectional LSTM using keras trained on the Brown Corpus

Paper used as reference - [Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Recurrent Neural Network](https://arxiv.org/pdf/1510.06168.pdf)

See [DetailedDescription.pdf](https://github.com/aneesh-joshi/LSTM_POS_Tagger/blob/master/DetailedDescription.pdf) for a detailed description of the whole project.


**Video Explanation:**
A video explaining the whole project can be found [here](https://drive.google.com/open?id=0B5-t3yDeHRzKVEZ4VUMwSWtwbDA)

## **TL;DR:**
The code does the following:

1. Extracts POS tagging training data from the Brown corpus (`extract_data.py`)
2. Converts a text file with the Glove Vectors into a pickle file (`make_glove_pickle.py`)
3. Trains a Bidirectional LSTM using the vectors and data. (`make_model.py`)
4. Allows pos tag prediction on new sentences fed in through `model_evaluation.py`

Uses keras and tensorflow backend

Glove file not included. It can be found [here](https://nlp.stanford.edu/projects/glove/)
Download `glove.6B.zip`
Unzip it and paste the `.txt`s in the current dir
Rest should be handled by the scripts

## Setup
Use `environment.yml` to set up the environment using anaconda

## Sample output of training (for 2 epochs):
```
(LSTM_POS_Tagger) D:\Projects\LSTM_POS_Tagger>python model_evaluation.py
Using TensorFlow backend.

The sentence is  ['i', 'want', 'to', 'dance', 'with', 'a', 'girl']
The tokenized sentence is  [[46187  7416  3956 31382 30171 28645 35332]]
The padded tokenized sentence is  [[    0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0     0     0     0     0     0 46187  7416  3956
  31382 30171 28645 35332]]
['i', 'want', 'to', 'dance', 'with', 'a', 'girl']
ppss
vb-hl
in-hl
vb
in
at
nn
```


## Example output on training
```
(LSTM_POS_Tagger) D:\Projects\LSTM_POS_Tagger>python make_model.py
Using TensorFlow backend.
TOTAL TAGS 471
TOTAL WORDS 49511
We have 36634 TRAINING samples
We have 9159 VALIDATION samples
We have 11449 TEST samples
Total 400000 word vectors.
Embedding matrix shape (49512, 100)
X_train shape (36634, 100)

model fitting - Bidirectional LSTM
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 100)               0
_________________________________________________________________
embedding_1 (Embedding)      (None, 100, 100)          4951200
_________________________________________________________________
bidirectional_1 (Bidirection (None, 100, 128)          84480
_________________________________________________________________
time_distributed_1 (TimeDist (None, 100, 472)          60888
=================================================================
Total params: 5,096,568
Trainable params: 5,096,568
Non-trainable params: 0
_________________________________________________________________
Epoch 1/2
1144/1144 [==============================] - 675s 590ms/step - loss: 0.2088 - acc: 0.9579 - val_loss: 0.0578 - val_acc: 0.9851
Epoch 2/2
1144/1144 [==============================] - 701s 613ms/step - loss: 0.0482 - acc: 0.9870 - val_loss: 0.0453 - val_acc: 0.9879
MODEL SAVED in Models/ as model.h5
TEST LOSS 0.043562
TEST ACCURACY: 0.987889
```