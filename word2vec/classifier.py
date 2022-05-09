from gensim.models import Word2Vec
import gensim
from knn import KNN
import numpy as np
from word2vec import Word2VecModel

class Classifier:
    def __init__(self, word2vec_weights):
        self.knn = KNN(5)
        #self.word2vec = Word2Vec.load(word2vec_weights)
        self.word2vec = Word2VecModel.load(word2vec_weights)

    def _get_average_word_embeddings(self, title):
        word_embeddings = [self.word2vec.wv[x] for x in title.split() if x in self.word2vec.wv]
        if len(word_embeddings) == 0:
            word_embeddings = np.zeros((1, 100))
        word_embeddings = np.array(word_embeddings)
        return np.mean(word_embeddings, axis=-2)
    
    def fit(self, X, y):
        training_vectors = [self._get_average_word_embeddings(x) for x in X]
        self.knn.fit(training_vectors, y)

    def predict(self, X):
        return self.knn.predict_numpy([self._get_average_word_embeddings(x) for x in X]).flatten()