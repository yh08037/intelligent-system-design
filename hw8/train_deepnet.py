# ID: 2018115809 (undergraduate)
# NAME: Dohun Kim
# File name: train_deepnet.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): os, sys, gzip, pickle, urllib, numpy=1.19.2, matplotlib=3.3.4

'''
train_deepnet.py :
    train DeepConvNet with MNIST / Fashion MNIST dataset
'''

############################## import required packages ##############################
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys, os, gzip, pickle

try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')

sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer
from matplotlib.offsetbox import AnchoredText



###################################### mnist #########################################

print('############### mnist ###############')

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



############################# define load_fashion_mnist ##############################

url_base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/fashion_mnist.pkl"

train_num = 60000
test_num = 10000
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist():
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")
    

def load_fashion_mnist(normalize=True, flatten=False):
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 



################################## fashion mnist #####################################

print('########### fashion mnist ###########')

(x_train, t_train), (x_test, t_test) = load_fashion_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

network = DeepConvNet()

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=10, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000,
                  verbose=False)

print('training DeepConvNet ...')
trainer.train()

# 매개변수 보관
network.save_params("params_fashion_mnist.pkl")
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
