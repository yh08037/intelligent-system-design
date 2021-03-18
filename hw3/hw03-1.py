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
    mlp = MLPClassifier(hidden_layer_sizes=(500,), max_iter=10, alpha=1e-4,
                        solver='sgd', verbose=False, random_state=1,
                        learning_rate_init=0.1)

    N_EPOCHS = 10
    BATCH_SIZE = 200
    CLASSES = np.unique(y_train)

    train_scores = []
    test_scores  = []

    epoch = 0
    while epoch < N_EPOCHS:
        for x, y in generate_batch(x_train, y_train, BATCH_SIZE):
            mlp.partial_fit(x, y, CLASSES)

        train_score = mlp.score(x_train, y_train)
        test_score  = mlp.score(x_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)

        print("epoch %d: train score = %f, test score = %f" %(epoch, train_score, test_score))
        
        epoch += 1
    
    if test_score[-1] > best_test_acc:
        best_test_acc = test_score[-1]
        best_hidden_layer_size = hidden_layer_size

    hidden_layer_size += 1

print('best hidden layer size: %d' %(hidden_layer_size))
print('best test accuracy: %f' %(best_test_acc))

# plt.plot(train_scores, alpha=0.8, label='Train')
# plt.plot(test_scores, alpha=0.8, label='Test')
# plt.title("Accuracy over epochs", fontsize=14)
# plt.xlabel('Epochs')
# plt.legend(loc='upper left')
# plt.show()
