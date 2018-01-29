#kfold tester


import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

X = np.matrix('1 2 3; 3 2 1; 3 2 2; 3 2 3; 1 2 2; 2 6 4')
y = np.matrix('3; 4; 1; 6; 2; 6')

kf = KFold(n_splits=3,shuffle=True)

for train_ind, test_ind in kf.split(X,y):
	print(X[train_ind])
	print(y[train_ind])
	print("fold!")