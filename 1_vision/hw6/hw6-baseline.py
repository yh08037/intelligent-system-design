# ID: 2018115809 (undergraduate)
# NAME: Dohun Kim
# File name: hw6-baseline.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): os, sys, numpy=1.19.2, matplotlib=3.3.4, scikit-learn=0.24.1

'''
hw6-baseline.py :
    test various data splitting methods with IRIS dataset
'''

############################## import required packages ##############################
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os, sys

from collections import OrderedDict
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.offsetbox import AnchoredText

sys.path.append(os.pardir)

from common.layers import Affine, Relu, SoftmaxWithLoss
from common.gradient import numerical_gradient


################################## define functions ##################################

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads



def train_neuralnet_iris(x_train, t_train, x_test, t_test, 
                         input_size=4, hidden_size=10, output_size=3, 
                         iters_num = 1000, batch_size = 10, learning_rate = 0.1,
                         verbose=True):
    
    network = TwoLayerNet(input_size, hidden_size, output_size)

    # Train Parameters
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size / batch_size, 1)

    train_loss_list, train_acc_list, test_acc_list = [], [], []

    for step in range(1, iters_num+1):
        # get mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 기울기 계산
        #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
        grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(압도적으로 빠르다)

        # Update
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # loss
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if step % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            
            if verbose:
                print('Step: {:4d}\tTrain acc: {:.5f}\tTest acc: {:.5f}'.format(step,
                                                                                train_acc,
                                                                                test_acc))
    
    tracc = network.accuracy(x_train, t_train)
    teacc = network.accuracy(x_test, t_test)

    if verbose:
        print('Optimization finished!')
        print('Training accuracy: %.2f' % tracc)
        print('Test accuracy: %.2f' % teacc)

    history = {
        'train_acc': train_acc_list,
        'test_acc': test_acc_list,
        'final_train_acc': tracc,
        'final_test_acc': teacc
    }

    return history



def plot_history(history, title):

    # plot learning curve by accuracy
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['test_acc'], label='test accuracy')

    # add test accuract in learning curve plot
    tracc = history['final_train_acc'] * 100
    teacc = history['final_test_acc'] * 100

    text = (f'final train acc: {tracc:.2f}%\n'
            + f'final test acc: {teacc:.2f}%')
    
    # anchored_text = AnchoredText(text, loc='lower center', frameon=False)
    anchored_text = AnchoredText(text, loc='lower center')
    plt.gca().add_artist(anchored_text)

    plt.legend(loc='lower right')

    # set axis
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.show()



def holdout_1(data_loader):

    '''Incorrect Holdout Split'''

    iris = data_loader()
    
    nsamples = iris.data.shape[0]
    
    ntestsamples = nsamples * 4 // 10  # `//' is integer division
    ntrainsamples = nsamples - ntestsamples	  # 4:6 test:train split

    testidx = range(0,ntestsamples)
    trainidx = range(ntestsamples,nsamples)

    history = train_neuralnet_iris(iris.data[trainidx,:], iris.target[trainidx],
                                   iris.data[testidx], iris.target[testidx],
                                   input_size=4, hidden_size=10, output_size=3, 
                                   iters_num = 1000, batch_size = 10, learning_rate = 0.1,
                                   verbose=False)

    plot_history(history, 'Incorrect Holdout Split')    



def holdout_2(data_loader):

    '''Per-class Holdout Split'''

    iris = data_loader()

    # 4:6 test:train split
    ntestsamples = len(iris.target) * 4 // 10  # '//' integer division
    ntestperclass = ntestsamples // 3

    # allocate indices for test and training data
    # Bte: boolean index for test data;  ~Bte: logical not, for training data
    Bte = np.zeros(len(iris.target),dtype=bool)   # initially, False index
    
    for c in range(0,3): 
        Bte[range(c*50,c*50+ntestperclass)] = True

    history = train_neuralnet_iris(iris.data[~Bte,:], iris.target[~Bte],
                                   iris.data[Bte,:], iris.target[Bte],
                                   input_size=4, hidden_size=10, output_size=3, 
                                   iters_num = 1000, batch_size = 10, learning_rate = 0.1,
                                   verbose=False)
    
    plot_history(history, 'Per-class Holdout Split')


def holdout_3(data_loader):

    '''Holdout Split by Random Sampling'''

    iris = data_loader()

    nsamples = iris.data.shape[0]
    ntestsamples = nsamples * 4 // 10  # 4:6 test:train split

    # random permutation (shuffling)
    Irand = np.random.permutation(nsamples)
    Ite = Irand[range(0,ntestsamples)]
    Itr = Irand[range(ntestsamples,nsamples)]

    history = train_neuralnet_iris(iris.data[Itr,:], iris.target[Itr],
                                   iris.data[Ite,:], iris.target[Ite],
                                   input_size=4, hidden_size=10, output_size=3, 
                                   iters_num = 1000, batch_size = 10, learning_rate = 0.1,
                                   verbose=False)

    plot_history(history, 'Holdout Split by Random Sampling')


def holdout_4(data_loader):
    
    '''sklearn.model_selection.train_test_split'''

    # We can now quickly sample a training set while holding out 
    # 40% of the data for testing (evaluating) our classifier:
    X, y = data_loader(return_X_y=True)
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.4, shuffle=True)

    history = train_neuralnet_iris(Xtr,ytr,Xte,yte,
                                   input_size=4, hidden_size=10, output_size=3, 
                                   iters_num = 1000, batch_size = 10, learning_rate = 0.1,
                                   verbose=False)

    plot_history(history, 'sklearn.model_selection.train_test_split')


    '''Improving Random Sampling Holdout'''

    X, y = data_loader(return_X_y=True)
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=len(y))

    # fix the SEED of random permutation to be the number of samples, 
    # to reproduce the same random sequence at every execution
    np.random.seed(len(y))

    history = train_neuralnet_iris(Xtr,ytr,Xte,yte,
                                   input_size=4, hidden_size=10, output_size=3, 
                                   iters_num = 1000, batch_size = 10, learning_rate = 0.1,
                                   verbose=False)
    
    plot_history(history, 'Improving Random Samplig Holdout')

    
def holdout_5(data_loader):

    '''Stratified Random Sampling'''

    X, y = data_loader(return_X_y=True)

    # per-class random sampling by passing y to variable stratify, 
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.4, shuffle=True, stratify=y)

    # check number of samples of the individual classes
    print('test: %d %d %d, '%(sum(yte==0),sum(yte==1),sum(yte==2)),end='')
    print('training: %d %d %d'%(sum(ytr==0),sum(ytr==1),sum(ytr==2)))

    # due to the random initialization of the weights, the performance varies
    # so we have to set the random seed for TwoLayerNet's initialization values
    np.random.seed(len(y))

    history = train_neuralnet_iris(Xtr,ytr,Xte,yte,
                                   input_size=4, hidden_size=10, output_size=3, 
                                   iters_num = 1000, batch_size = 10, learning_rate = 0.1,
                                   verbose=False)

    plot_history(history, 'Stratified Random Sampling')



def crossval_1(data_loader):
    
    '''Repeated Random Subsampling'''
    # Repeating stratified random sampling K times

    X, y = data_loader(return_X_y=True)

    # due to the random initialization of the weights, the performance varies
    # so we have to set the random seed for TwoLayerNet's initialization values
    np.random.seed(len(y))

    K = 20
    Acc = np.zeros([K,2], dtype=float)
    
    for k in range(K):
        # stratified random sampling
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=None, stratify=y)
        
        history = train_neuralnet_iris(Xtr,ytr,Xte,yte,
                                       input_size=4, hidden_size=10, output_size=3, 
                                       iters_num = 1000, batch_size = 10, learning_rate = 0.1, 
                                       verbose = False)
        
        tracc = history['final_train_acc']
        teacc = history['final_test_acc']
        
        print('Trial %d: accuracy %.3f %.3f' % (k, tracc, teacc))


def crossval_2(data_loader):

    '''LOO (leave-one-out) cross validation, k nearest neighbors on IRIS'''
    # LOO is useul for KNN, because no model training is required

    iris = data_loader()
    for k in [1,3,5,7,9]:
        neigh = KNeighborsClassifier(n_neighbors=k)
        I = np.ones(iris.target.shape,dtype=bool)   # True index array
        y_pred = -np.ones(iris.target.shape,dtype=int)    # prediction, assigned -1 for initial values
        for n in range(len(iris.target)):
            I[n] = False    # unselect, leave one
            y_pred[n] = neigh.fit(iris.data[I,:], iris.target[I]).predict(iris.data[n,:].reshape(1,-1))
            I[n] = True     # select, for the next step

        nsamples = iris.data.shape[0]
        nmisses = (iris.target != y_pred).sum()
        print('kNN with k=%d' % k)
        print('Number of mislabeled out of a total %d samples : %d (%.2f%%)'
                % (nsamples, nmisses, float(nmisses)/float(nsamples)*100.0))



######################################## main ########################################

if __name__ == '__main__':

    funcs = [
        (holdout_1, True),
        (holdout_2, True),
        (holdout_3, True),
        (holdout_4, True),
        (holdout_5, True),
        (crossval_1, True),
        (crossval_2, True)
    ]

    data_loader = load_iris

    for f, use_data_loader in funcs:
        print('running ' + f.__name__ + '() ...')
        if use_data_loader:
            f(data_loader)
        else:
            f()
        print()
