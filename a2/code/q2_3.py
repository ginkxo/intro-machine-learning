'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''

    trainBinary = binarize_data(train_data)

    # ~ WE HAVE all the data
    # ~ first : divide it based on class labels
    # ~ second : for elements 0 to 63 in the matrix, in class k, nkj is 
    # ~         just the sum of 1's in the column over all points

    eta = np.zeros((10, 64))

    for k in range(0,10):
        digitsClassK = data.get_digits_by_label(trainBinary, train_labels, k)
        digitsShape = np.shape(digitsClassK)
        numPoints = digitsShape[0]
        numFeats = digitsShape[1]
        for j in range(0,numFeats):
            featSum = np.sum(digitsClassK[:,j])
            eta[k,j] = (featSum + 1) / (numPoints + 2) 
            # ~ + 1 and + 2 equivalent to adding two data points, one all 0s and one all 1s
            # ~ from naive bayes beta distro slides, equivalent to MAP estimation prior


    
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    etas = []
    for i in range(10):
        img_i = class_images[i]
        img8by8 = np.reshape(img_i,(8,8))
        etas.append(img8by8)

    all_concat = np.concatenate(etas, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''

    numFeats = np.shape(eta)[1]

    generated_data = np.zeros((10, 64))

    for k in range(0,10):
        for j in range(0,numFeats):
            etaJK = eta[k,j]
            generated_data[k,j] = generate_point(etaJK)

    plot_images(generated_data)

# ~ helper function
def generate_point(eta_jk):
    etaJK = int(round(100 * eta_jk))
    rand = np.random.randint(1,100)
    if rand <= etaJK:
        return 1
    else:
        return 0

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''

    numDigits, numFeats = np.shape(bin_digits)[0], np.shape(bin_digits)[1]
    genLikelihood = np.zeros((numDigits,10))

    for b in range(numDigits):
        for k in range(0,10):
            pB = 1
            for j in range(numFeats):
                etaKJ = eta[k,j]
                bJ = bin_digits[b,j]
                pB = pB * ((etaKJ)**(bJ))*((1 - etaKJ)**(1-bJ))
            genLikelihood[b,k] = pB

    genLikelihood = np.log(genLikelihood)

    return genLikelihood 

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    pY = 1./10
    pyVec = np.zeros((10,1))
    pyVec.fill(pY)

    cond = generative_likelihood(bin_digits,eta)
    condexp = np.exp(cond)
    normal = np.matmul(condexp,pyVec)

    condlike = (cond + np.log(pY)) - np.log(normal) 

    return condlike

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return
    numDigits = np.shape(bin_digits)[0]
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

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    numDigits = np.shape(bin_digits)[0]

    predictions = np.zeros((numDigits,1))

    for i in range(numDigits):
        predictions[i,0] = np.argmax(cond_likelihood[i])

    return predictions 

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
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    #plot_images(eta)

    generate_new_data(eta)

    avgCondTrain = avg_conditional_likelihood(train_data, train_labels, eta)
    avgCondTest = avg_conditional_likelihood(test_data, test_labels, eta)

    classifyTrain = classify_data(train_data, eta)
    classifyTest = classify_data(test_data, eta)

    accTrain = test_accuracy(train_labels, classifyTrain)
    accTest = test_accuracy(test_labels, classifyTest)

    print(avgCondTrain, avgCondTest)
    print(accTrain, accTest)



if __name__ == '__main__':
    main()
