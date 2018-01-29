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
 
 
#to implement
def LRLS2(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    # for each w* we want to calculate, we have to compute A ourselves
    # the size of A will be the dimensions of the number of features squared
    # we need to compute A each time! and for each diagonal element, we need to compute the a^i
    ## TODO

    # ASSUMPTIONS:
        # test_datum is a MATRIX
        # 

    Ntrain = x_train.shape[0]
    A = np.zeros((Ntrain, Ntrain))
    A2 = np.zeros((Ntrain, Ntrain))

    td_tr = test_datum.transpose() # 1 x d
    # POSSIBLY NEED TO TURN IT TO MATRIX IF DOESNT WORK
    # x_train is Ntrain x d
    # each point: x_train[i,:]
    # dists = l2(td_tr, x_train)
    # result: dists is a 1 x Ntrain matrix where:
      # dists[0, i] = ||td_tr[0,:] - x_train[i,:]||^2

    dists = l2(td_tr, x_train) # gives us a i: 0->Ntrain-1 sized 1 x Ntrain matrix
    # each element i : ||td_tr[0,:] - x_train[i,:]||^2 aka ||x - x^(i)||^2

    div_dists = (-1.0)*dists/(2*(tau)**2)
   # B = np.max(div_dists)
    exp_dists = np.exp(div_dists)
    normalizer = np.exp(logsumexp(div_dists))
    #normalizer = np.sum(exp_dists)

    rowd,cold = np.diag_indices(A.shape[0])
    A[rowd,cold] = exp_dists/normalizer

    X_tr_b = np.insert(x_train, 0, 1, axis=1) 

    xtay = np.matmul(X_tr_b.transpose(), np.matmul(A, y_train))
    xtax = np.matmul(X_tr_b.transpose(), np.matmul(A, X_tr_b)) 

    a, b = xtax.shape 
    I = np.identity(a)
    reg = lam * I
    toinvert = xtax + reg
    Xsqinv = np.linalg.solve(toinvert, I)

    wstar = np.matmul(Xsqinv, xtay)
    test_bias = np.insert(td_tr,0,1,axis=1)
    y_pred = np.matmul(test_bias, wstar)

    #print(y_pred)
    return y_pred[0]
    ## TODO

def accumulate(foldx):
  fold_array = list(foldx)
  N = len(fold_array)
  while N != 1:
    #x = zeros(fold_array[0].shape)
    #x[:,:] = fold_array[0]
    tmp = np.concatenate((fold_array[0], fold_array[1]), axis=0)
    fold_array[1] = tmp
    dummy = fold_array.pop(0)
    N = len(fold_array)
    #print(fold_array[0])
  return fold_array[0]

def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
  # made an identity matrix with lambda in diagonals of length x_train[0]
  # made a diagonal matrix of Ntrain x Ntrain with the diagonals containing -(1.0)/(2*tau**2)
  # numerators -> we np.diag a matrix, this matrix is the np.matmul result of l2(test_datum.transpose(), x_train) and t
  #   and then we np.diag(np.matmul(a,b)[0])

  # denominator -> math.exp of logsumexp of numerators
  # denominators -> an np.diag of 1.0/denominator for x in range Ntrain

  # for each numerators[x][x] for x from 0 to Ntrain, we set numerators[x][x] to be math.exp(numerators[x][x])
  # A is just np.matmul(numerators, denominators)
  # w = np.linalg.solve(np.matmul(np.matmul(x_train.transpose(), A), x_train) + I, np.matmul(np.matmul(x_train.transpose(), A), y_train))
  # return w

  rows, cols = x_train.shape 
  I = np.diag(np.array([lam for i in range(cols)]))
  tauscale = (-(1.0)/(2*tau**2))
  const_tau_diags = np.diag(np.array([tauscale for i in range(rows)]))

  base_dist = l2(test_datum.transpose(), x_train)
  num_dist_consts = np.matmul(base_dist, const_tau_diags)
  top = np.diag(num_dist_consts[0])

  bot = np.exp(logsumexp(top))
  normalization = np.diag([(1.0)/bot for x in range(rows)])

  #for x in range(rows):
  #  top[x][x] = np.exp(top[x][x])

  top_diag = np.diagonal(top)
  top_diag_exp = np.exp(top_diag)
  top = np.diag(top_diag_exp)

  A = np.matmul(top,normalization)
  
  w = np.linalg.solve(np.matmul(np.matmul(x_train.transpose(), A), x_train) + I, np.matmul(np.matmul(x_train.transpose(), A), y_train))

  # np.dot(w, test_datum)
  
  #test_bias = np.insert(test_datum,0,1,axis=0)
  y_pred = np.matmul(w, test_datum)
    #print(y_pred)
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


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    #Xa = np.matrix()
    #Yb = np.matrix()
    #taus = np.logspace(1.0, 3, 10)
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    #print(losses)
    plt.plot(taus, losses)
    plt.title('normal')
    plt.xlabel('tau points')
    plt.ylabel('loss value')
    plt.grid(True)
    plt.show()

    print("min loss = {}".format(losses.min()))

