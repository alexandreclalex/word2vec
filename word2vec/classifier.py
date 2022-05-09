from word2vec import Word2VecModel
from gensim.models import Word2Vec
from knn import KNN
import numpy as np


class Classifier:
    def __init__(self, word2vec_weights):
        # Create the KNN model and load a pretrained word2vec instance
        self.knn = KNN(5)
        self.word2vec = Word2VecModel()
        self.word2vec.load(word2vec_weights)


    def _get_average_word_embeddings(self, title):
        '''
        For each title, fetch the word2vec embedding for each word. Then, average the embeddings
        for one title embedding
        @param title: The title for which to fetch the word embedding
        '''
        word_embeddings = [self.word2vec.predict(x) for x in title.split()]
        if len(word_embeddings) == 0:
            word_embeddings = np.zeros((1, self.word2vec.projection_dim))
        word_embeddings = np.array(word_embeddings)
        me = np.mean(word_embeddings, axis=0).flatten()
        return me


    def fit(self, X, y):
        '''
        Fit the KNN model based on average word embeddings
        
        @param X: The titles
        @param y: The labels
        '''
        training_vectors = [self._get_average_word_embeddings(x) for x in X]
        self.knn.fit(training_vectors, y)


    def predict(self, X):
        '''
        Predict the class based on the word embeddings

        @param X: The titles to use in the predictions
        '''
        return self.knn.predict_numpy([self._get_average_word_embeddings(x) for x in X]).flatten()