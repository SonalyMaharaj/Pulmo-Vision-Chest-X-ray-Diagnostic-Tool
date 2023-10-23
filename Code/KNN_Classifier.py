from collections import defaultdict
import numpy as np

def most_frequent(List):
    '''Function to find the most frequent element in a list, and the confidence of the prediction'''
    occurence_count = defaultdict(int)
    for item in List:
        occurence_count[item] += 1
    max_value = max(occurence_count.values())
    max_keys = [k for k, v in occurence_count.items() if v == max_value]
    return max_keys[0], max_value/len(List)

global YtestLen # length of test data

class KNearestNeighbor:
    ''' Implements the KNearest Neigbours For Classification... '''
    def __init__(self, k, scalefeatures=False):        
        self.k = k # number of nearest neighbours
        self.scalefeatures = scalefeatures  # whether to scale features or not

    def compute_distances_no_loops(self, X):
        """ Compute the distance between each test point in X and each training point in self.X_train using no explicit loops. """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))  # initialize distance matrix as zero matrix
        dists  = np.sum(np.square(X), axis=1, keepdims=True) + np.sum(np.square(self.X_train), axis=1) - 2 * np.dot(X, self.X_train.T) # compute distance matrix
        return dists
    
    def scale_features(self,X):
        """Normalize each feature to lie in the range [0 ,1] """
        self.xmin = np.min(X,axis=0) # minimum of each column
        self.xmax = np.max(X,axis=0) # maximum of each column
        Normalized_X = (X - self.xmin) / (self.xmax - self.xmin) # normalize
        return Normalized_X # return normalized data

    def train(self, X, Y):
        ''' Train K Nearest Neighbour classifier using the given Train Data '''
        if self.scalefeatures:
            X = self.scale_features(X)
        self.X_train = X # store the train data
        self.Y_train = Y # store the train labels
    
    def predict(self, X):
        """ Test the trained K-Nearset Neighoubr classifier result on the given examples X """
        num_test = X.shape[0] # number of test data
        
        if self.scalefeatures: # scale the features if required
            X = (X-self.xmin)/(self.xmax-self.xmin)
        
        y_pred = np.zeros(num_test, dtype = self.Y_train.dtype) # initialize y_pred as zero vector
        compute_distance = self.compute_distances_no_loops # compute distance between each test point and training point
        dists = compute_distance(X) #dists is the distance matrix between each test point and training point

        # find the k nearest neighbours for each test point
        y_pred_classes = []
        for idx, data in enumerate(dists):
            y_pred_classes.append(self.Y_train[np.argsort(data)[:self.k]])

        # find the most frequent class among the k nearest neighbours
        y_pred = []
        y_confidence = []
        for i in y_pred_classes:
            pred,confidence = most_frequent(list(i))
            y_pred.append(pred)
            y_confidence.append(confidence)

        return y_pred, y_confidence