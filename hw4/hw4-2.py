# ID: 2018115809 (undergraduate)
# NAME: Dohun Kim
# File name: hw4-2.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): sys, os, gzip, numpy=1.19.2 scikit-learn=0.24.1

'''
hw4-2.py :
    classification of scikit-learn digit dataset with `TwoLayerNet`
'''

############################## import required packages ##############################
# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from sklearn.datasets import load_digits
from two_layer_net import TwoLayerNet


################################### preparing data ###################################

# load data from gzip files
x, t = load_digits(return_X_y=True)

# normalization of input data
x = x / x.max()

# split dataset into training(80%) and test(20%) sets
train_rate = 0.8
num_label = np.unique(t, axis=0).shape[0]

train_idx, test_idx = [], []

for i in range(num_label):
    all_idx = list(np.where(t == i)[0])
    num_train = int(len(all_idx) * train_rate)

    train_idx += all_idx[:num_train]
    test_idx  += all_idx[num_train:]

x_train, t_train = x[train_idx], t[train_idx]
x_test,  t_test  = x[test_idx],  t[test_idx]

# check the proportion of the result data
print('label train   test')
for i in range(num_label):
    a = len(t_train[t_train==i])
    b = len(t_test[t_test==i])
    print('  %1d   %3.2f%%  %2.2f%%' %(i, a/(a+b)*100, b/(a+b)*100))

# one-hot encoding
num_label = np.unique(t_train, axis=0).shape[0]
t_train = np.eye(num_label)[t_train]
t_test  = np.eye(num_label)[t_test]


################################# train and test model ################################

network = TwoLayerNet(input_size=64, hidden_size=50, output_size=10)

iters_num = 500
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(int(train_size / batch_size), 1)

print(len(x), iter_per_epoch)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('iter %-5d\ttrain_acc: %-3.2f%%\ttest_acc: %-3.2f%%' %(i, train_acc*100, test_acc*100))
