'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        # ~ this will depend on the value of k
        # ~ if k == 1, we just take the point with the minimum distance, check its label, and assign it the same label
        # ~ if k > 1, we take the k closest points, and vote on the label. If there is a tie, tiebreak somehow (randomly, for now?)

        # ~ digit = None
        # ~ return digit

        # ~ return index for the k smallest distances
        # ~ get the labels for these indices
        # ~ vote
        # ~ return the digit of the majority vote

        minIndices = []
        trainDistances = self.l2_distance(test_point) # ~ may be able to delete
        dynamicMins = self.l2_distance(test_point)
        fakeInfty = np.max(trainDistances) + 999;

        for i in range(k):
            ix = np.argmin(dynamicMins)
            minIndices.append(ix)
            dynamicMins[ix] = fakeInfty

        assert len(minIndices) == k

        labels = [self.train_labels[ind] for ind in minIndices]
        # ~ return max(set(labels), key=labels.count)

        return self.choose_label(labels)

    def choose_label(self, labels):
        '''
        ~ Custom helper function
        Take an array of label values and return the voted-on label.

        Tiebreaking: 
        A tie occurs if there is more than one maximum integer in the internal labelFreqs array.
        Based on np.argmax, the tie is broken by choosing the lowest index.
        '''

        labelFreqs = np.zeros(10)

        for label in labels:
            lint = int(label)
            labelFreqs[lint] = labelFreqs[lint] + 1

        return float(np.argmax(labelFreqs))



def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    kf = KFold(n_splits=10,shuffle=True)

    accuracies = []

    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        accuraciesK = []
        for trainDataIdx, testDataIdx in kf.split(train_data,train_labels):
            trainingSet = train_data[trainDataIdx]
            trainingLabels = train_labels[trainDataIdx]
            testSet = train_data[testDataIdx]
            testLabels = train_labels[testDataIdx]
            knn = KNearestNeighbor(trainingSet, trainingLabels)
            classAccuracy = classification_accuracy(knn, k, testSet, testLabels)
            accuraciesK.append(classAccuracy)
        accuraciesK = np.array(accuraciesK)
        accuraciesKMean = np.mean(accuraciesK)
        accuracies.append(accuraciesKMean)
        # print(accuracies)

    accuracies = np.array(accuracies)
    return np.argmax(accuracies) + 1 # this is the K with the highest accuracy



def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    # ~ pass

    # ~ take a knn object and k value
    # ~ for each point in eval_data:
    # ~     call knn.query_knn(eval_data[i], k)
    # ~     check output prediction with eval_labels[i]
    # ~     if correct, set errorVector[i] to 1
    # ~     accuracy -> sum and divide by length of array

    numPts = eval_data.shape[0]
    errorVector = np.zeros(numPts)

    for i in range(numPts):
        evalPt = eval_data[i]
        evalLabel = eval_labels[i]
        pred = knn.query_knn(evalPt, k)
        if (pred == evalLabel):
            errorVector[i] = 1

    return (np.sum(errorVector)/numPts)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    #predicted_label = knn.query_knn(test_data[0], 1)
    #print(predicted_label)

    # k_one = classification_accuracy(knn, 1, test_data, test_labels) # 0.96875
    # k_fifteen = classification_accuracy(knn, 15, test_data, test_labels) # 0.9585
    # print(k_one, k_fifteen)

    goodK = cross_validation(train_data, train_labels)
    print(goodK)
    k_good = classification_accuracy(knn, goodK, test_data, test_labels)
    k_good2 = classification_accuracy(knn, goodK, train_data, train_labels)
    print(k_good2)
    print(k_good)

    for i in range(1,16):
        kg = classification_accuracy(knn, i, train_data, train_labels)
        kg2 = classification_accuracy(knn, i, test_data, test_labels)
        print(kg, kg2)

    # results:
    #   [
    #       0.9649
    #       0.9579
    #       0.9649
    #       0.9599
    #       0.9624
    #       0.9602
    #       0.9582
    #       0.9552
    #       0.9524
    #       0.9537
    #       0.9529
    #       0.9506
    #       0.9509
    #       0.9511
    #       0.9487
    #   ]



if __name__ == '__main__':
    main()