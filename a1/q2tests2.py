from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
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
    A_norm = (np.square(A)).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (np.square(B)).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
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

    td_tr = test_datum.transpose() # 1 x d
    # POSSIBLY NEED TO TURN IT TO MATRIX IF DOESNT WORK
    # x_train is Ntrain x d
    # each point: x_train[i,:]
    # dists = l2(td_tr, x_train)
    # result: dists is a 1 x Ntrain matrix where:
      # dists[0, i] = ||td_tr[0,:] - x_train[i,:]||^2

    dists = l2(td_tr, x_train) # gives us a i: 0->Ntrain-1 sized 1 x Ntrain matrix
    # each element i : ||td_tr[0,:] - x_train[i,:]||^2 aka ||x - x^(i)||^2

    div_dists = dists/(-(2*tau)**2)
    B = np.max(div_dists)
    exp_dists = np.exp(div_dists - B)
    normalizer = np.sum(exp_dists)

    #for i in range(Ntrain):
    #  A[i][i] = exp_dists[0,i]/normalizer
    rowd,cold = np.diag_indices(A.shape[0])
    A[rowd,cold] = exp_dists 

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

    return y_pred[0,0]

def kfold(x,k):
    N = x.shape[0]
    i = 0
    folds = []
    single_fold_size = int(np.ceil(N/k))
    while i < N:
        folds.append(x[i:i+single_fold_size,:])
        i = i + single_fold_size

    return folds 

def split_fold(x, k):
    N = x.shape[0]
    folds = []
    split = 1.0/k*N
    for i in range(k):
        folds.append(x[int(round(i*split)):int(round((i+1)*split)), :])
    return folds

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

    #lossarr = np.zeros(k, taus.shape)
    for i in range(k):
        x_test = folds[i][:,0:-1]
        y_test = folds[i][:,-1]
        excluded_fold = folds.pop(i)
        xy_train = accumulate(folds)
        x_train = xy_train[:,0:-1]
        y_train = xy_train[:,-1]
        #lossarr[i,:] = run_on_fold(x_test, y_test, x_train, y_train, taus)
        print(taus)
        print(x_test)
        print(y_test)
        print(x_train)
        print(y_train)
        folds.insert(i, excluded_fold)
        print("new iteration! item: ", i)

    ## TODO
    return None


if __name__ == "__main__":
    '''
    A = np.matrix('1 2 3 4 5')
    a = A.transpose() 
    B = np.matrix('2 4 8 4 3; 2 3 5 3 2; 6 3 4 5 3')
    y = np.matrix('4; 6; 3')
    t = 1.0
    '''

    X = np.matrix('1 2 1; 1 2 2; 3 2 3; 2 3 2; 3 2 1; 7 9 4; 3 3 2')
    y = np.matrix('3; 2; 1; 2; 7; 0; 8')

    run_k_fold(X, y, 1, 4)

    