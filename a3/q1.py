'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import time

import pickle

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def tf_idf_features2(bow_train, bow_test):
    tf_idf_tnf = TfidfTransformer()
    tf_idf_train = tf_idf_tnf.fit_transform(bow_train)
    tf_idf_test = tf_idf_tnf.transform(bow_test)
    return tf_idf_train, tf_idf_test


def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def svm(train_d, train_labels, test_d, test_labels):

    # alpha 1e-4 results in 0.95 test acc 0.68 train acc

    model = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=5)

    fitting_time = time.time()

    model.fit(train_d, train_labels)

    print("Fitting time: %s seconds" % (time.time() - fitting_time))

    prediction_time = time.time()

    train_pred = model.predict(train_d)
    test_pred = model.predict(test_d)

    print("Prediction time: %s seconds" % (time.time() - prediction_time))

    print('Basic SVM train accuracy = {}'.format((train_pred == train_labels).mean()))
    print('Basic SVM test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model 

def linearsvc(train_d, train_labels, test_d, test_labels):

    model = LinearSVC(penalty='l2',loss='hinge',C=1.0,multi_class='ovr')

    fitting_time = time.time()

    model.fit(train_d, train_labels)

    print("Fitting time: %s seconds" % (time.time() - fitting_time))

    prediction_time = time.time()

    train_pred = model.predict(train_d)
    test_pred = model.predict(test_d)

    print("Prediction time: %s seconds" % (time.time() - prediction_time))

    print('Basic SVM train accuracy = {}'.format((train_pred == train_labels).mean()))
    print('Basic SVM test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model   


def randomforest(train_d, train_labels, test_d, test_labels):

    model = RandomForestClassifier(n_estimators=100)

    fitting_time = time.time()

    model.fit(train_d, train_labels)

    print("Fitting time: %s seconds" % (time.time() - fitting_time))

    prediction_time = time.time()

    train_pred = model.predict(train_d)
    test_pred = model.predict(test_d)

    print("Prediction time: %s seconds" % (time.time() - prediction_time))

    print('Basic SVM train accuracy = {}'.format((train_pred == train_labels).mean()))
    print('Basic SVM test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model      

def multilogreg(train_d, train_labels, test_d, test_labels):

    model = LogisticRegression(penalty='l2', C=1.0, solver='newton-cg', multi_class='multinomial')

    fitting_time = time.time()

    model.fit(train_d, train_labels)

    print("Fitting time: %s seconds" % (time.time() - fitting_time))

    prediction_time = time.time()

    train_pred = model.predict(train_d)
    test_pred = model.predict(test_d)

    print("Prediction time: %s seconds" % (time.time() - prediction_time))

    print('Basic MultiLogReg train accuracy = {}'.format((train_pred == train_labels).mean()))
    print('Basic MultiLogReg test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model   

def gnb(train_d, train_labels, test_d, test_labels):

    model = GaussianNB()

    fitting_time = time.time()

    model.fit(train_d, train_labels)

    print("Fitting time: %s seconds" % (time.time() - fitting_time))

    prediction_time = time.time()

    train_pred = model.predict(train_d)
    test_pred = model.predict(test_d)

    print("Prediction time: %s seconds" % (time.time() - prediction_time))

    print('Basic GaussianNB train accuracy = {}'.format((train_pred == train_labels).mean()))
    print('Basic GaussianNB test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model   

def mnb(train_d, train_labels, test_d, test_labels):

    model = MultinomialNB(alpha=0.01)

    fitting_time = time.time()

    model.fit(train_d, train_labels)

    print("Fitting time: %s seconds" % (time.time() - fitting_time))

    prediction_time = time.time()

    train_pred = model.predict(train_d)
    test_pred = model.predict(test_d)

    print("Prediction time: %s seconds" % (time.time() - prediction_time))

    print('Basic MultinomialNB train accuracy = {}'.format((train_pred == train_labels).mean()))
    print('Basic MultinomialNB test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def confusion_matrix(pred_y, real_y):
    '''
    .: Generates a confusion matrix for the 20 Newsgroups classes.

    inputs ::
        (pred_y) < test prediction N-vector 
        (real_y) < test data target N-vector 

    outputs ::
        (confusion) < 20 x 20 confusion matrix

    '''

    confusion = np.zeros((20,20))
    N = pred_y.shape

    assert N == real_y.shape # ensure alignment of dimensions

    for i in range(N[0]):
        realClass = real_y[i]
        predictedClass = pred_y[i]
        confusion[predictedClass,realClass] += 1

    return confusion   


if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    train_tfidf, test_tfidf, feature_names_tfidf = tf_idf_features(train_data, test_data)
    train_tfidf2, test_tfidf2 = tf_idf_features2(train_bow, test_bow)

    print(test_data.target.shape)
    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)

    # svm_model = svm(train_tfidf2, train_data.target, test_tfidf2, test_data.target)
    mnb_model = mnb(train_tfidf2, train_data.target, test_tfidf2, test_data.target)
    testpred = mnb_model.predict(test_tfidf2)
    conf = confusion_matrix(testpred, test_data.target)
    np.savetxt("confusion.csv", conf, delimiter=",", fmt='%.1e')

    # rf_model = randomforest(train_tfidf2, train_data.target, test_tfidf2, test_data.target)

    # filename = 'svm.sav'
    # pickle.dump(svm_model, open(filename, 'wb'))

    # random forest training accuracy = 0.9747
    # random forest test accuracy = 0.5916

    # Gaussian NB training accuracy = --
    # Gaussian NB test accuracy = --

    # Multinomal NB training accuracy = 0.9589
    # Multinomal NB test accuracy = 0.7002

    # SVM training accuracy = 0.9540
    # SVM test accuracy = 0.6966

    # MultiLogReg training accuracy = 0.9073
    # MultiLogReg test accuracy = 0.6737

    # bernoulliNB training accuracy = 0.5988
    # bernoulliNB test accuracy = 0.4579

    #bnbsav = 'bnb.sav'
    #pickle.dump(bnb_mode, open(bnbsav, 'wb'))

    # mlr_model = multilogreg(train_tfidf, train_data.target, test_tfidf, test_data.target)
    # print(mlr_model.get_params())

    
    '''
    mlr_param_grid = [
        {'C': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], 'solver': ['newton-cg']},
        {'C': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], 'solver': ['sag']},
        {'C': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], 'solver': ['saga']},
        {'C': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], 'solver': ['lbfgs']}

    ]
    '''


    '''
    mnb_param_grid = [
        {'alpha': [1.0, 0.5, 0.1, 0.01, 0.001]}
    ]
    '''






    
