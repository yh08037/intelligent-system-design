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
from matplotlib.offsetbox import AnchoredText

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

# create MLP model object
mlp = MLPClassifier(hidden_layer_sizes=(500,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=True, random_state=1,
                    learning_rate_init=0.1)

# train MLP model and ignore convergence warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                            module="sklearn")
    mlp.fit(x_train, y_train)


##################################### test model #####################################

# get final score of trained MLP model in percent
train_score = mlp.score(x_train, y_train) * 100
test_score  = mlp.score(x_test, y_test) * 100

# plot training loss over 10 epochs
plt.plot(mlp.loss_curve_, alpha=0.8, label='training loss')
plt.title("Training Loss over epochs", fontsize=14)
plt.xlabel('Epochs')

# add test accuract in learning curve plot
score_string = 'train score : %.2f%%\ntest score : %.2f%%' %(train_score, test_score)
anchored_text = AnchoredText(score_string, loc='upper right')
plt.gca().add_artist(anchored_text)

plt.show()


################################ plot example filters ################################

# define row, column number of subplot
row, col = 5, 5

# get weights of 0th hidden layer
coefs = mlp.coefs_[0].T

# get index of 25 filters which have highest stddev values
coef_std = coefs.std(axis=1)
idx = np.argsort(coef_std)[-row*col:]

# use global min / max to ensure all weights are shown on the same scale
coef_min, coef_max = coefs.min(), coefs.max()

fig, axes = plt.subplots(row, col)

for coef, ax in zip(coefs[idx], axes.ravel()):
    ax.matshow(coef.reshape(8, 8), cmap=plt.cm.gray, 
               vmin=coef_min*0.5, vmax=coef_max*0.5)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
