import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1849)

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

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        ## return None

        self.vel = -1 * self.lr * grad + (self.beta * self.vel)
        return params + self.vel


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.matrix(np.random.normal(0.0, 0.1, feature_count+1)) 
        self.t = 1 # iteration number t
        self.w[0,0] = 1
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        n = X.shape[0]
        # Implement hinge loss
        ## return None

        ###

        ### IMPORTANT NOTE: FOR X and feature_count, we assume BIAS IS INCLUDED IN X. SO KEEP THIS IN MIND

        ###

        X_transp = np.transpose(X) # (n,m) -> (m,n) bc each wtX is w0x0 + w1x1 + ... + wmxm
        wtX = np.matmul(self.w, X_transp) # (1,m) x (m,n) 
        out_w = np.transpose(wtX) # (n,1) dimension

        for i in range(n):
            curroutw = out_w[i,:]
            out_w[i,:] = y[i] * curroutw

        basehinge = 1 - out_w
        hinge = np.matrix(np.zeros((n,2)))

        hinge[:,0] = basehinge[:,0]

        return hinge.max(axis=1)

    def gradcheck(self, y, wtx, x):

        featnum = x.shape[0]
        zeros = np.zeros(featnum)

        if y*wtx >= 1:
            return zeros
        else:
            return -1 * y * x

    def grad(self, X, y):

        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''

        current_w = self.w

        numPts = X.shape[0]
        penaltFtr = self.c / numPts

        for i in range(numPts):
            currX = X[i,:]
            currY = y[i]
            wTx = np.matmul(self.w, currX)
            grad = self.gradcheck(currY, wTx, currX)
            current_w = current_w + penaltFtr * grad 

        return current_w

 
    

    def gradPegasos(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        
        # Compute (sub-)gradient of SVM objective

        # our input is already the gradient batch
        m = X.shape[0]
        nonzeroX = []
        nonzeroY = []

        yloss = self.hinge_loss(X, y)

        for i in range(m):
            if yloss[i] > 0:
                nonzeroX.append(X[i,:])
                nonzeroY.append(y[i])

        Atp = np.vstack(nonzeroX)
        yAtp = np.vstack(nonzeroY)
        eta_t = 1. / self.t

        nAtp = Atp.shape[0]
        dAtp = Atp.shape[1]


        extsum = np.zeros((1,dAtp))

        for j in range(nAtp):
            extsum = extsum + (yAtp[j] * Atp[j,:])


        w_t_half = (1 - eta_t)*self.w + (self.c * eta_t)/(m) * extsum 

        wt1 = min(1., 1./(np.linalg.norm(w_t_half))) * w_t_half

        return wt1


    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1

        xn = X.shape[0]
        y = np.zeros(xn)

        for i in range(xn):
            currentDataPt = X[i,:]
            dpT = np.transpose(currentDataPt)
            score = np.dot(self.w, dpT)
            if (score >= 0):
                y[i] = 1.0
            else:
                y[i] = -1.0

        return y       

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for i in range(steps):
        # Optimize and update the history
        ## pass
        new_w = optimizer.update_params(w_history[i], func_grad(w_history[i]))
        w_history.append(new_w) 
    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    featcount = train_data.shape[1]

    bs = BatchSampler(train_data, train_targets, batchsize)
    svm = SVM(penalty, featcount-1)

    for i in range(iters):

        Xb, yb = bs.get_batch()
        out_w = svm.grad(Xb, yb)
        svm.t = svm.t + 1
        new_w = optimizer.update_params(svm.w, out_w)
        svm.w = new_w

    return svm

if __name__ == '__main__':


    '''
    
    GD = GDOptimizer(1.0)
    GDM = GDOptimizer(1.0,0.9)

    w1 = optimize_test_function(GD)
    w2 = optimize_test_function(GDM)

    w1 = np.array(w1)
    w2 = np.array(w2)

    f1 = 0.01 * w1 * w1
    f2 = 0.01 * w2 * w2

    rng = np.arange(1.,202.,1)

    plt.plot(rng, w1, 'r--', rng, w2, 'b--')
    plt.xlabel("Iteration")
    plt.ylabel("w value")
    plt.show()

    plt.plot(rng, f1, 'r--', rng, f2, 'b--')
    plt.xlabel("Iteration")
    plt.ylabel("f(w) value")
    plt.show()

    '''

    print("Getting data ...")

    X_train, y_train, X_test, y_test = load_data()

    print("Data extracted.")
    print("Creating optimizers ...")

    GD = GDOptimizer(0.05)
    GDM = GDOptimizer(0.05, 0.1)

    C = 1.0
    bSize = 100
    T = 500

    print("Optimizers ready.")
    print("Adding biases ...")


    X_train_b = np.insert(X_train, 0, 1, axis=1)
    # y_train_b = np.insert(y_train, 0, 1, axis=0)

    X_test_b = np.insert(X_test, 0, 1, axis=1)
    # y_test_b = np.insert(y_test, 0, 1, axis=0)

    print("Biases added.")
    print("Optimizing SVMs ...")

    svm1 = optimize_svm(X_train_b, y_train, C, GD, bSize, T)
    svm2 = optimize_svm(X_train_b, y_train, C, GDM, bSize, T)

    print("SVMs trained.")

    
    TA_hinge_loss_1 = svm1.hinge_loss(X_train_b, y_train) # turn this into an array over all losses?
    TA_hinge_loss_2 = svm2.hinge_loss(X_train_b, y_train) # turn this into an array over all losses?

    TE_hinge_loss_1 = svm1.hinge_loss(X_test_b, y_test)
    TE_hinge_loss_2 = svm2.hinge_loss(X_test_b, y_test) 

    TA_average_hinge_loss_1 = np.mean(TA_hinge_loss_1)
    TA_average_hinge_loss_2 = np.mean(TA_hinge_loss_2)

    TE_average_hinge_loss_1 = np.mean(TE_hinge_loss_1)
    TE_average_hinge_loss_2 = np.mean(TE_hinge_loss_2)

    print("TA1 average hinge-loss: ", TA_average_hinge_loss_1)
    print("TA2 average hinge-loss: ", TA_average_hinge_loss_2)
    print("TE1 average hinge-loss: ", TE_average_hinge_loss_1)
    print("TE2 average hinge-loss: ", TE_average_hinge_loss_2)

    y_train_out_1 = svm1.classify(X_train_b)
    y_train_out_2 = svm2.classify(X_train_b)

    y_out_1 = svm1.classify(X_test_b)
    y_out_2 = svm2.classify(X_test_b)

    train_accuracy_1 = y_train_out_1 == y_train
    train_accuracy_2 = y_train_out_2 == y_train 

    ta_1_total = train_accuracy_1.shape[0]
    ta_2_total = train_accuracy_2.shape[0]

    ta_1 = np.sum(train_accuracy_1) / ta_1_total
    ta_2 = np.sum(train_accuracy_2) / ta_2_total

    test_accuracy_1 = y_out_1 == y_test
    test_accuracy_2 = y_out_2 == y_test

    te_1_total = test_accuracy_1.shape[0]
    te_2_total = test_accuracy_2.shape[0]

    te_1 = np.sum(test_accuracy_1) / te_1_total
    te_2 = np.sum(test_accuracy_2) / te_2_total


    print("TA 1: ", ta_1)
    print("TA 2: ", ta_2)
    print("TE 1: ", te_1)
    print("TE 2: ", te_2)

    w1sq = np.reshape(svm1.w[0,1:], (28,28))
    w2sq = np.reshape(svm2.w[0,1:], (28,28))

    sqs = [w1sq, w2sq]
    cnct = np.concatenate(sqs, 1)
    plt.imshow(cnct, cmap='gray')
    plt.show()


    






