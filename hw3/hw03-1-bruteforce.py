# ID: 2018115809 (undergraduate)
# NAME: Dohun Kim
# File name: hw03-1-bruteforce.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): os, gzip, numpy=1.19.2 pandas=1.2.3 
#                      matplotlib=3.3.4 scikit-learn=0.24.1

'''
hw03-1-bruteforce.py :
    get optimal hidden layer size for hw03-1.py via brute-force search
'''

############################## import required packages ##############################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

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

def generate_batch(x: np.array, y: np.array, batch_size):
    assert x.shape[0] == y.shape[0]

    for idx in range(0, x.shape[0], batch_size):
        yield x[idx:idx+batch_size], y[idx:idx+batch_size]


best_test_acc = 0
best_hidden_layer_size = 1
hidden_layer_size = 1

while hidden_layer_size <= 1000:

    print('============== hidden layer size : %d ==============' %(hidden_layer_size))

    mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), max_iter=10, alpha=1e-4,
                        solver='sgd', verbose=False, random_state=1,
                        learning_rate_init=0.1)

    # this example won't converge because of CI's time constraints, so we catch the
    # warning and are ignore it here
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        mlp.fit(x_train, y_train)

    train_score = mlp.score(x_train, y_train)
    test_score  = mlp.score(x_test, y_test)

    print('train score = %f, test score = %f' %(train_score, test_score))
    
    f = open('./score.log', 'a')
    f.write('%d,%f,%f\n' %(hidden_layer_size, train_score, test_score))
    f.close()

    if test_score > best_test_acc:
        best_test_acc = test_score
        best_hidden_layer_size = hidden_layer_size

    hidden_layer_size += 1

    print()

print('best hidden layer size: %d' %(best_hidden_layer_size))
print('best test accuracy: %f' %(best_test_acc))
