# ID: 2018115809 (undergraduate)
# NAME: Dohun Kim
# File name: hw8-1.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): os, sys, numpy=1.19.2, matplotlib=3.3.4

'''
hw8-1.py :
    test DeepConvNet with MNIST dataset
'''

############################## import required packages ##############################
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer
from matplotlib.offsetbox import AnchoredText



################################# train_deepnet.py ###################################

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

network = DeepConvNet()  

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=10, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.002},
                  evaluate_sample_num_per_epoch=1000,
                  verbose=False)

print('training DeepConvNet ...')
trainer.train()

# 매개변수 보관
network.save_params("params_mnist.pkl")
print("Saved Network Parameters!")

# plot learning curve
plt.plot(trainer.train_acc_list, label='train acc')
plt.plot(trainer.test_acc_list, label='test acc')
plt.title('Accuracy - train vs test')
plt.legend()

# add final accuract in learning curve plot
tracc = trainer.train_acc_list[-1] * 100
teacc = trainer.test_acc_list[-1] * 100

text = (f'final train acc: {tracc:.2f}%\n' + f'final test acc: {teacc:.2f}%')

anchored_text = AnchoredText(text, loc='lower center')
plt.gca().add_artist(anchored_text)
plt.show()


############################## misclassified_mnist.py ################################

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params("params_mnist.pkl")

print("calculating test accuracy ... ")
#sampled = 1000
#x_test = x_test[:sampled]
#t_test = t_test[:sampled]

classified_ids = []

acc = 0.0
batch_size = 100

for i in range(int(x_test.shape[0] / batch_size)):
    tx = x_test[i*batch_size:(i+1)*batch_size]
    tt = t_test[i*batch_size:(i+1)*batch_size]
    y = network.predict(tx, train_flg=False)
    y = np.argmax(y, axis=1)
    classified_ids.append(y)
    acc += np.sum(y == tt)
    
acc = acc / x_test.shape[0]
print("test accuracy:" + str(acc))

classified_ids = np.array(classified_ids)
classified_ids = classified_ids.flatten()
 
max_view = 20
current_view = 1

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)

mis_pairs = {}
for i, val in enumerate(classified_ids == t_test):
    if not val:
        ax = fig.add_subplot(4, 5, current_view, xticks=[], yticks=[])
        ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')

        plt.title(f'{t_test[i]} -> {classified_ids[i]}')
        mis_pairs[current_view] = (t_test[i], classified_ids[i])
            
        current_view += 1
        if current_view > max_view:
            break

plt.suptitle('misclassified results')
plt.tight_layout()

# print("======= misclassified result =======")
# print("{view index: (label, inference), ...}")
# print(mis_pairs)

plt.show()
