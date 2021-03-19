# ID: 2018115809 (undergraduate)
# NAME: Dohun Kim
# File name: hw03-2.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): os, gzip, numpy=1.19.2 pandas=1.2.3 
#                      matplotlib=3.3.4 scikit-learn=0.24.1

'''
hw03-2.py :
    classification of 8x8 digit dataset with MLPClassifier
'''

############################## import required packages ##############################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import warnings
from sklearn.exceptions import ConvergenceWarning


################################### preparing data ###################################

# load digits dataset
x, y = load_digits(return_X_y=True)

# normalization of input data
x = x / x.max()

# split dataset into training(80%) and test(20%) sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                    random_state=0)


##################################### train model ####################################

mlp = MLPClassifier(hidden_layer_sizes=(500,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=True, random_state=1,
                    learning_rate_init=0.1)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                            module="sklearn")
    mlp.fit(x_train, y_train)

    train_score = mlp.score(x_train, y_train)
    test_score  = mlp.score(x_test, y_test)

    print('train score = %f, test score = %f' %(train_score, test_score))


fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(8, 8), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

