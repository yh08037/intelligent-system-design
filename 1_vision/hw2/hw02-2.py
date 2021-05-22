# ID: 2018115809 (undergraduate)
# NAME: Dohun Kim
# File name: hw02-2.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): numpy=1.19.2 pandas=1.2.3 
#                      matplotlib=3.3.4 scikit-learn=0.24.1

'''
hw02-2.py :
    classification of versicolor vs virginica
'''

############################## import required packages ##############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from matplotlib.offsetbox import AnchoredText


############################## define perceptron class ###############################

class Perceptron:
    """
    Perceptron neuron
    """

    def __init__(self, learning_rate=0.1):
        """
        instantiate a new Perceptron

        :param learning_rate: coefficient used to tune the model
        response to training data
        """
        self.learning_rate = learning_rate
        self._b = 0.0  # y-intercept
        self._w = None  # weights assigned to input features
        # count of errors during each iteration
        self.misclassified_samples = []

    def fit(self, x: np.array, y: np.array, n_iter=10):
        """
        fit the Perceptron model on the training data

        :param x: samples to fit the model on
        :param y: labels of the training samples
        :param n_iter: number of training iterations 
        """
        self._b = 0.0
        self._w = np.zeros(x.shape[1])
        self.misclassified_samples = []

        for _ in range(n_iter):
            # counter of the errors during this training iteration
            errors = 0
            for xi, yi in zip(x, y):
                # for each sample compute the update value
                update = self.learning_rate * (yi - self.predict(xi))
                # and apply it to the y-intercept and weights array
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)

            self.misclassified_samples.append(errors)

    def f(self, x: np.array) -> float:
        """
        compute the output of the neuron
        :param x: input features
        :return: the output of the neuron
        """
        return np.dot(x, self._w) + self._b

    def predict(self, x: np.array):
        """
        convert the output of the neuron to a binary output
        :param x: input features
        :return: 1 if the output for the sample is positive (or zero),
        -1 otherwise
        """
        return np.where(self.f(x) >= 0, 1, -1)


################################# download iris data #################################

# download and convert the csv into a DataFrame
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)


################################### preparing data ###################################

# define target label name list
target_label = ['Iris-versicolor', 'Iris-virginica']
target_name = ['Versicolor', 'Virginica']

# select rows from DataFrame based on label column values
df = df.loc[df[4].isin(target_label)]

x = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

# map the labels to a binary integer value
# 1: versicolor, -1: virginica
y = np.where(y==target_label[0], 1, -1)

# standardization of the input features
x_means = x.mean(axis=0)
x_stds = x.std(axis=0)
x = (x - x_means) / x_stds

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9,
                                                    random_state=0)
                                                    

##################################### train model ####################################

# train the model
classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train, n_iter=5)

# plot the number of errors during each iterations
plt.plot(range(1, len(classifier.misclassified_samples) + 1),
         classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.title('Learning curve: ' + target_name[0] + '-' + target_name[1])


##################################### test model #####################################

# test the model
prediction = classifier.predict(x_test)
accuracy = (prediction==y_test).sum() / len(y_test) * 100

# add test accuract in learning curve plot
anchored_text = AnchoredText(f'test accuracy: {accuracy:.2f}%', loc='upper right')
plt.gca().add_artist(anchored_text)
plt.show()