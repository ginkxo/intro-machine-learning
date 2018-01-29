from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as met 

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        xfeat = np.squeeze(np.asarray(X[:,i]))
        plt.plot(xfeat, y, 'ro')
        plt.ylabel("Value")
        plt.xlabel(features[i])
    
    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    X = np.insert(X, 0, 1, axis=1) # adds the bias value, 
    m, n = X.shape # m rows, n columns
    # now a m x d + 1 dimension matrix

    Xsq = np.matmul(X.transpose(), X) # n x n
    Xty = np.matmul(X.transpose(), Y) # n x m x n x 1

    i1, i2 = Xsq.shape 
    I = np.identity(i1)

    Xsqinv = np.linalg.solve(Xsq, I) 

    W = np.matmul(Xsqinv, Xty) 
    return W 


def main():
    # Load the data
    X, y, features = load_data()

    # --- added ---

    dim = np.size(y)
    # --- added ---

    #print(features)

    print("Features: {}".format(features))
    

    # Visualize the features
    visualize(X, y, features)

    #TODO: Split data into train and test
    rows, cols = X.shape 
    Xs = np.insert(X, cols, 1, axis=1)
    Xs[:,-1] = y # design:target

    np.random.shuffle(Xs) # randomizes design:target 
    eighty = int(dim*0.8) # approx 80 percent of data
    train, test = Xs[:eighty,:], Xs[eighty:,:] # split 80-20

    ytr = train[:,-1]
    yts = test[:,-1]

    train = train[:,0:-1]
    test = test[:,0:-1]

    # Fit regression model
    w = fit_regression(train, ytr) # 13 dims + bias 

    # Compute fitted values, MSE, etc.

    print(w) 

    testb = np.insert(test, 0, 1, axis=1)  # include bias element 
    ypred = np.matmul(testb, w) # prediction 
    msqe = met.mean_squared_error(yts, ypred) # mean squared error, see writeup
    msle = met.mean_squared_log_error(yts, ypred) # mean sq log error, see writeup
    mae = met.mean_absolute_error(yts, ypred) # mean abs error, see writeup

    print(msqe)
    print(msle)
    print(mae)

# ENTRY POINT: 
if __name__ == "__main__":
    main()
