# LSTM_POS_Tagger
A simple POS Tagger made using a Bidirectional LSTM using keras trained on the Brown Corpus

See DetailedDescription.pdf for a detailed description of the whole project.

**TL;DR:**
The code does the following:

1. Extracts POS tagging training data from the Brown corpus (`extract_data.py`)
2. Converts a text file with the Glove Vectors into a pickle file (`make_glove_pickle.py`)
3. Trains a Bidirectional LSTM using the vectors and data. (`make_model.py`)
4. Allows pos tag prediction on new sentences fed in through `model_evaluation.py`

Uses keras and tensorflow backend

Glove file not included