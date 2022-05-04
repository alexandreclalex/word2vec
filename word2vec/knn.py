# %load knn.py
import math
import random
import numpy as np
from scipy import spatial
from scipy import stats

class KNN:
    def __init__(self, k):
        """
        Takes one parameter. k is the number of nearest neighbors to use
        to predict the output variable's value for a query point.
        """
        self.k = k

    def fit(self, X, y):
        self.reference_points = X
        self.reference_values = y

    def predict_loop(self, X):
        # 1d list of predictions. Each prediction in the list
        # corresponds to the value of the input array at the same index.
        predictions = []
        for q in X:
            # a list of tuples in format (distance_from_reference, reference_
            sorted_neighbors = []
            for index, reference_point in enumerate(self.reference_points):
                intermediate_dist_sum = 0
                for i in range(len(reference_point)):
                    intermediate_dist_sum += (q[i] - reference_point[i]) ** 2
                    distance = math.sqrt(intermediate_dist_sum)
                    sorted_neighbors.append((distance, self.reference_values[index]))
                    sorted_neighbors.sort(key=lambda x: x[0])
                    knns = sorted_neighbors[0: self.k]
                    prediction = max(set([x[1] for x in knns]), key=[x[1] for x in knns])
                    predictions.append(prediction)
        return np.array(predictions)

    def predict_numpy(self, X):
        # This line creates a distance matrix. The matrix contains a row for
        # and contains a column for every reference point. The values of this
        # from the input point of the row to every reference point. The input
        # containing points to be classified.
        distances = spatial.distance.cdist(X, self.reference_points, 'euclidean')
        # The line below takes the distance matrix as described above and use
        # function on each row. This method sorts the data in ascending order
        # return a sorted version of the matrix, it returns a matrix with ind
        # where the values of the row should be if sorted. This matrix is the
        # top <k> neighbors for each row. The resulting matrix is a <input_le
        # with the indexes of the top k nearest references to the input point
        knn_indexes = np.argsort(distances)[:, :self.k]
        # The below line uses numpy's take method to get the known values for
        # input matrix containing the indices of the top <k> values per row.
        # these values in order to determine the most common value for the to
        # the resulting array is flattened to produce a one dimensional outpu
        return np.array([stats.mode(np.take(self.reference_values, x))[0] for x in knn_indexes])