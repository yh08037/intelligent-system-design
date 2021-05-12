# ID: 2018115809 (undergraduate)
# NAME: Dohun Kim
# File name: hw8-2.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): os, sys, gzip, pickle, urllib, numpy=1.19.2, matplotlib=3.3.4

'''
hw8-2.py :
    test DeepConvNet with Fashion MNIST dataset
'''

############################## import required packages ##############################
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os, sys, gzip, pickle

try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')

sys.path.append(os.pardir)

from deep_convnet import DeepConvNet
from common.trainer import Trainer
from matplotlib.offsetbox import AnchoredText



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



################################# train_deepnet.py ###################################

(x_train, t_train), (x_test, t_test) = load_fashion_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

network = DeepConvNet()  
network.load_params("params_fm_small_10.pkl")

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


############################## misclassified_mnist.py ################################

(x_train, t_train), (x_test, t_test) = load_fashion_mnist(flatten=False)

network = DeepConvNet()
network.load_params("params_fashion_mnist.pkl")

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
