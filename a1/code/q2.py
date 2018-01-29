# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.misc import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''

    print("In fold.")

    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
        print("loss value: ", j)

    print("Returning losses ...")
    return losses
 
# self-defined helper function
def accumulate(foldx):
    '''
    Takes an initial set of K - 1 folds as an array and combines them into a single training matrix.  
    '''
    fold_array = list(foldx)
    N = len(fold_array)
    while N != 1:
        tmp = np.concatenate((fold_array[0], fold_array[1]), axis=0)
        fold_array[1] = tmp
        dummy = fold_array.pop(0)
        N = len(fold_array)
    return fold_array[0]

# self-defined helper function
def randomize_data(X,y):
    '''
    Input:  X is an N x d design matrix of data points
            y is the d x 1 target vector
    Output: X and y of same dimensions, but randomized order for the data pts
    '''

    # combine X and y into a single matrix Xs
    # ensures that targets stick with their data points
    rows, cols = X.shape 
    Xs = np.insert(X, cols, 1, axis=1)
    # y_mat = np.matrix(y)
    Xs[:,-1] = y # Xs data pt 1 = [x1 ... xd|y] 

    np.random.shuffle(Xs) # shuffle data entries

    # separate X and y again
    y_out_mat = Xs[:,-1]
    y_out_mat_t = y_out_mat.transpose()
    y_out = np.squeeze(np.asarray(y_out_mat_t))
    X_out = Xs[:,0:-1]

    return X_out, y_out



# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    rows, cols = x_train.shape # Ntrain, numfeats (13, occasionally + 1 bias)
    I = np.diag(np.array([lam for i in range(cols)])) # numfeats x numfeats matrix where diagonals are lambda, 
    #equiv to lambda * I

    tauscale = (-(1.0)/(2*tau**2)) # -1/2tau^2
    const_tau_diags = np.diag(np.array([tauscale for i in range(rows)])) # Ntrain x Ntrain matrix where diagonals 
    #are the tau constant

    base_dist = l2(test_datum.transpose(), x_train) # l2 distance -> ||x - x_train(i)||^2 for each i 
    num_dist_consts = np.matmul(base_dist, const_tau_diags) 
    top = np.diag(num_dist_consts[0]) # numfeats x numfeats matrix of 
    # ||x - x_train(i)||^2 divided by -1/2tau^2 as diagonals

    bot = np.exp(logsumexp(top)) 
    normalization = np.diag([(1.0)/bot for x in range(rows)]) # matrix with normalization constant as diag entries

    top_diag = np.diagonal(top) # Makes a diagonal matrix of the ||x - x_train(i)||^2 / -1/2tau^2 terms as diag entries
    top_diag_exp = np.exp(top_diag) # Matrix cloned with diag entries now exp'd
    top = np.diag(top_diag_exp) # re-assign top as the same matrix but with exp( ||x - x_train(i)||^2 / -1/2tau^2 ) diags

    A = np.matmul(top,normalization) # forms the A matrix with A_ii defined as the a_i weights in assignment 

    # solving for inversion as per 2.1 and calculation of w vector
    XtA = np.matmul(x_train.transpose(), A)

    XtAX = np.matmul(XtA, x_train)
    XtAy = np.matmul(XtA, y_train)

    w = np.linalg.solve(XtAX + I, XtAy)

    y_pred = np.matmul(w, test_datum)
    return y_pred[0]


def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''

    # 1 . split the data into k folds

    xy = np.insert(x, x.shape[1], 1, axis=1)
    xy[:,-1] = y 

    N = x.shape[0]
    folds = []
    split = 1.0/k*N
    for i in range(k):
        folds.append(xy[int(round(i*split)):int(round((i+1)*split)), :])
    
    # folds now contains the design matrix rows (data pts) divided into folds

    lossarr = np.zeros((k, taus.shape[0]))
    for i in range(k):
        x_test = folds[i][:,0:-1]
        y_test = folds[i][:,-1]
        excluded_fold = folds.pop(i)
        xy_train = accumulate(folds)
        x_train = xy_train[:,0:-1]
        y_train = xy_train[:,-1]
        print("Running fold ...")
        lossarr[i,:] = run_on_fold(x_test, y_test, x_train, y_train, taus)
        folds.insert(i, excluded_fold)
        print("k fold: ", k)
    ## TODO
    taumeans = np.mean(lossarr, axis=0)
    return taumeans 
    ## TODO


# ENTRY POINT:
if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)

    x_rand, y_rand = randomize_data(x,y)
    losses = run_k_fold(x_rand,y_rand,taus,k=5)

    plt.plot(taus, losses)
    plt.title('normal')
    plt.xlabel('tau points')
    plt.ylabel('loss value')
    plt.grid(True)
    plt.show()

    print("min loss = {}".format(losses.min()))

