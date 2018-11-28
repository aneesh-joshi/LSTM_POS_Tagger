import nltk
from collections import Counter

class DataLoader:
    """
    1. Loads the brown corpus
    2. Tokenizes the sentences and tags
    3. Gets the word2index and tog2index mappings
    """
    def __init__(self):
        # Make sure nltk is installed and brown corpus is downloaded
        nltk.download('brown')
        self.sentences = nltk.corpus.brown.tagged_sents()
        self.extract_data()

    def extract_data(self):
        self.num_sents = len(self.sentences)
        self.tokenized_sentence = []
        self.tokenized_tags = []
        self.word2index, self.index2word = {}, {}
        self.tag2index, self.index2tag = {}, {}
        wordCounter, tagCounter = Counter(), Counter()
        for sentence in self.sentences:
            this_sentence, this_sentence_tags = [], []
            for (word, tag) in sentence:
                wordCounter.update(word)
                tagCounter.update(tag)
                # this_sentence.append()
                # this_sentence_tags.append(tag)
        print(wordCounter)


if __name__ == '__main__':
    DataLoader()