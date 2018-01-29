import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch    


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''

    # see the formula in the assignment writeup
    dim = len(y)
    grad_c = 2.0/dim
    wt = w.transpose()
    accumulator = 0
    for i in range(dim):
        accumulator += - y[i] + np.matmul(wt,X[i])
    lr_grad_w = np.multiply(grad_c * accumulator, wt)
    return lr_grad_w

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # Example usage
    X_b, y_b = batch_sampler.get_batch()
    batch_grad = lin_reg_gradient(X_b, y_b, w)

    # ==== part 3.5 ====

    gradL = lin_reg_gradient(X, y, w) # grad of whole L
    dimgradL = gradL.shape
    gradL_s = np.zeros(dimgradL)

    K = 500
    m = 50
    invK = (1.)/(K)

    # experimental formula as defined in 3.5
    # we sample K batches of size m
    # we continually add up the resulting gradient, gradL_s
    # and then divide by K 

    for i in range(K):
        Xsample, ysample = batch_sampler.get_batch(m)
        outgradL_s = lin_reg_gradient(Xsample, ysample, w)
        gradL_s = gradL_s + outgradL_s 

    gradL_s = invK * gradL_s 

    # cosine similarity and squared distance 

    cosine_sim = cosine_similarity(gradL, gradL_s)
    squared_dist = np.linalg.norm(gradL - gradL_s)**2

    print("cosine similarity: ", cosine_sim)
    print("squared distance: ", squared_dist)

    # cosine more meaningful!

    # ==== part 3.6 =====

    Msize = 400
    wsize = w.shape[0]

    # range of our m values [1,400]
    M = np.linspace(1, 400, num=Msize)

    # calculation of variance vector:
    # for each m in [1,400] we construct a gradients matrix of 14 x K size
    # and then we sample batches of size m K times, calculate the gradient, and 
    # put each gradient vector as entry gradients[:,j] 
    # finally, we manually calculate the variance per each weight in this gradients matrix

    variances = np.zeros((wsize,Msize))
    for i in range(Msize):
        gradients = np.zeros((wsize,K))
        for j in range(K):
            Xsmp, ysmp = batch_sampler.get_batch(int(M[i]))
            outgradls = lin_reg_gradient(Xsmp, ysmp, w)
            # this will give a single gradient vector
            gradients[:,j] = outgradls 


    # manual calculation of variances 

        gradmeans = np.mean(gradients, axis=1)
        gradmeans = np.matrix(gradmeans).transpose()
        gradsubmean = gradients - gradmeans
        entrysq = np.square(gradsubmean)
        gradsum = np.sum(entrysq, axis=1)
        gradvars = (1./(K))*gradsum
        variances[:,i] = gradvars.flatten()



    # log weight variacnes vs. log m
    # we arbitrarily choose w1 

    plt.plot(np.log(M), np.log(variances[1,:]))
    plt.title("log weight variance vs. m")
    plt.xlabel("log m")
    plt.ylabel("log weight variance")
    plt.show()


# ENTRY POINT:
if __name__ == '__main__':
    main()