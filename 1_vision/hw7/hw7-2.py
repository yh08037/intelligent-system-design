# ID: 2018115809 (undergraduate)
# NAME: Dohun Kim
# File name: hw7-2.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): os, sys, numpy=1.19.2, matplotlib=3.3.4

'''
hw7-2.py :
    test LeNet with MNIST dataset
'''

############################## import required packages ##############################
# coding: utf-8
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.pardir)

from collections import OrderedDict
from matplotlib.image import imread
from matplotlib.offsetbox import AnchoredText

from common.layers import *
from common.gradient import numerical_gradient
from common.trainer import Trainer
from dataset.mnist import load_mnist



############################# define class LeNet #############################

class LeNet:
    ''' LeNet
    conv - relu - pool - conv - relu - pool - affine - relu 
      - affine - relu - affine - softmax 
    
    input_size  = (1, 28, 28)
    output_size = 10
    hidden_size = [120, 84]
    '''
    def __init__(self, weight_init_std=0.01):

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(6, 1, 5, 5)
        self.params['b1'] = np.zeros(6)
        
        self.params['W2'] = weight_init_std * np.random.randn(16, 6, 5, 5)
        self.params['b2'] = np.zeros(16)

        self.params['W3'] = weight_init_std * np.random.randn(400, 120)
        self.params['b3'] = np.zeros(120)

        self.params['W4'] = weight_init_std * np.random.randn(120, 84)
        self.params['b4'] = np.zeros(84)
        
        self.params['W5'] = weight_init_std * np.random.randn(84, 10)
        self.params['b5'] = np.zeros(10)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['C1'] = Convolution(self.params['W1'], self.params['b1'], pad=2)
        self.layers['Relu1'] = Relu()
        self.layers['S2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['C3'] = Convolution(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['S4'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Relu4'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W5'], self.params['b5'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3, 4, 5):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['C1'].dW, self.layers['C1'].db
        grads['W2'], grads['b2'] = self.layers['C3'].dW, self.layers['C3'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W5'], grads['b5'] = self.layers['Affine3'].dW, self.layers['Affine3'].db

        return grads
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['C1', 'C3', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]



#################################### train LeNet #####################################

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = LeNet()
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000,
                  verbose=False)

print("Training...")
trainer.train()

# 매개변수 보존
network.save_params("lenet_params.pkl")
print("Saved Network Parameters!")

# 그래프 그리기
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title('Accuracy - train vs test')

# add final accuract in learning curve plot
tracc = trainer.train_acc_list[-1] * 100
teacc = trainer.test_acc_list[-1] * 100

text = (f'final train acc: {tracc:.2f}%\n'
        + f'final test acc: {teacc:.2f}%')

anchored_text = AnchoredText(text, loc='lower center')
plt.gca().add_artist(anchored_text)

plt.show()



################################## visualize filter ##################################

def filter_show(filters, nx=4, show_num=16):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(show_num / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(show_num):
        ax = fig.add_subplot(3, 3, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')


network = LeNet()
# 무작위(랜덤) 초기화 후의 가중치
filter_show(network.params['W1'], show_num=6)
plt.suptitle('Randomly initialized weights')
plt.tight_layout()
plt.show()

# 학습된 가중치
network.load_params("lenet_params.pkl")
filter_show(network.params['W1'], show_num=6)
plt.suptitle('Trained weights')
plt.tight_layout()
plt.show()



#################################### apply filter ####################################

img = imread('../dataset/cactus_gray.png')
img = img.reshape(1, 1, *img.shape)

fig = plt.figure()

w_idx = 1

for i in range(6):
    w = network.params['W1'][i]
    b = 0  # network.params['b1'][i]

    w = w.reshape(1, *w.shape)
    #b = b.reshape(1, *b.shape)
    conv_layer = Convolution(w, b) 
    out = conv_layer.forward(img)
    out = out.reshape(out.shape[2], out.shape[3])
    
    ax = fig.add_subplot(3, 3, i+1, xticks=[], yticks=[])
    ax.imshow(out, cmap=plt.cm.gray_r, interpolation='nearest')

plt.suptitle('Filter applied image')
plt.show()

