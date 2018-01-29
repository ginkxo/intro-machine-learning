'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means

    # ~ 0 0 0 0 0 0 0 0
    # ~ 0 0 0 0 0 0 0 0

    # ~ we have a 10 x 64 matrix
    # ~ we have 10 entries, each a 64-entry vector, corresponding to each class

    for i in range(0,10):
        iDigits = data.get_digits_by_label(train_data, train_labels, i)
        meanDigits = np.mean(iDigits,axis=0)
        means[i,:] = meanDigits 

    # ~ 10 x 64 matrix -> each row corresponds to a digit
    # ~ since each image is 8 x 8, we have it flattened into a 1 x 64 
    # ~ so each row is the mean values of all digits corresponding to that class

    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances

    # ~ covariance matrix is a 64 x 64 matrix, and we have 10 of these, one per class 
    # ~ so 10 x 64 x 64 is our return 
    # ~

    meanVals = compute_mean_mles(train_data, train_labels)

    for i in range(0,10):
        iDigits = data.get_digits_by_label(train_data, train_labels, i)
        for j in range(0,64):
            x1 = np.transpose(iDigits[:,j])
            for k in range(0,64):
                x2 = np.transpose(iDigits[:,k])
                covariances[i,j,k] = cov(x1,x2,meanVals[i,j],meanVals[i,k])

        epsilon = 0.01 * np.identity(64) # ~ stability
        covariances[i,:,:] = covariances[i,:,:] + epsilon


    return covariances

# ~ helper function
def cov(x1, x2, x1mean, x2mean):

    assert np.shape(x1)[0] == np.shape(x2)[0]

    size = np.shape(x1)[0]

    x1m = x1 - x1mean
    x2m = np.transpose(x2 - x2mean)
    numSum = np.dot(x1m,x2m)

    cov = numSum / size

    return cov 

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    covs = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        covDiag8by8 = np.reshape(cov_diag,(8,8))
        covs.append(covDiag8by8)

    all_concat = np.concatenate(covs, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''

    numDigits = np.shape(digits)[0]
    generate = np.zeros((numDigits,10))

    for i in range(numDigits):
        x = np.transpose(digits[i]) # ~ could edit ut later if messy
        d = np.shape(x)[0]
        #print(d)
        for k in range(0,10):
            meanVector = np.transpose(means[k]) # ~ could edit out letter if messy
            covMatrix = covariances[k]
            covMatrixInv = np.linalg.inv(covMatrix)
            normal = x - meanVector
            mahalanobis = 0.5 * np.matmul(np.matmul(np.transpose(normal),covMatrixInv),normal)
            probK = ((2*np.pi)**(-d/2))*((np.linalg.det(covariances[k]))**(-0.5))*np.exp(-mahalanobis)

            generate[i,k] = probK

    generate = np.log(generate)

    return generate

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    pY = 1./10
    pyVec = np.zeros((10,1))
    pyVec.fill(pY)

    cond = generative_likelihood(digits,means,covariances)
    condexp = np.exp(cond)
    normal = np.matmul(condexp,pyVec)

    condlike = (cond + np.log(pY)) - np.log(normal) 

    return condlike

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    numDigits = np.shape(digits)[0]
    avgNum = []

    for x in range(numDigits):
        yTrue = int(labels[x])
        yCorrectCond = cond_likelihood[x,yTrue]
        avgNum.append(yCorrectCond)

    avgNum = np.array(avgNum)

    avgSum = np.sum(avgNum)
    avgSumNorm = avgSum / numDigits 

    # Compute as described above and return
    return avgSumNorm

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    numDigits = np.shape(digits)[0]

    predictions = np.zeros((numDigits,1))

    for i in range(numDigits):
        predictions[i,0] = np.argmax(cond_likelihood[i])

    return predictions 

# ~ helper function
def test_accuracy(labels, predictions):

    assert np.shape(labels)[0] == np.shape(predictions)[0]

    numDigits = np.shape(predictions)[0]
    hits = 0

    for i in range(numDigits):
        iLabel = int(labels[i])
        iPred = predictions[i,0]
        if iLabel == iPred:
            hits = hits + 1

    accuracy = hits / numDigits 

    return accuracy



def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means_tr = compute_mean_mles(train_data, train_labels)
    covariances_tr = compute_sigma_mles(train_data, train_labels)

    means_te = compute_mean_mles(test_data, test_labels)
    covariances_te = compute_sigma_mles(test_data, test_labels)

    # print(covariances[0,:,:])

    # Evaluation

    avg_train = avg_conditional_likelihood(train_data, train_labels, means_tr, covariances_tr)
    avg_test = avg_conditional_likelihood(test_data, test_labels, means_tr, covariances_tr)
    
    preds_tr = classify_data(train_data, means_tr, covariances_tr)
    preds_te = classify_data(test_data, means_te, covariances_te)
    preds_te_r = classify_data(test_data, means_tr, covariances_tr)

    train_acc = test_accuracy(train_labels, preds_tr)
    # test_acc = test_accuracy(test_labels, preds_te)
    test_acc_r = test_accuracy(test_labels, preds_te_r)

    # plot_cov_diagonal(covariances)

    print(avg_train, avg_test)
    print(train_acc, test_acc_r)

if __name__ == '__main__':
    main()