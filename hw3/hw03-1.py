# ID: 2018115809 (undergraduate)
# NAME: Dohun Kim
# File name: hw03-1.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): os, gzip, numpy=1.19.2 pandas=1.2.3 
#                      matplotlib=3.3.4 scikit-learn=0.24.1

'''
hw03-1.py :
    classification of Fashion MNIST dataset with MLPClassifier
'''

############################## import required packages ##############################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from matplotlib.offsetbox import AnchoredText

import warnings
from sklearn.exceptions import ConvergenceWarning


############################# define data loader function ############################

def load_fashion_mnist(path, kind='train'):
    '''
    original code is found at:
    https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    '''
    import os
    import gzip

    """Load Fashion MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


################################### preparing data ###################################

# load data from gzip files
x_train, y_train = load_fashion_mnist('./', 'train')
x_test,  y_test  = load_fashion_mnist('./', 't10k')

# normalization of input data
x_train = x_train / 255.
x_test  = x_test  / 255.


##################################### train model ####################################

# hidden layer size: obtained by brute-force search from 1 to 1000
hidden_layer_size = 645

# create MLP model object
mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), max_iter=10, alpha=1e-4,
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
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, 
               vmin=coef_min*0.5, vmax=coef_max*0.5)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
