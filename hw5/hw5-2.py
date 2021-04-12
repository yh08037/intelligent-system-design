# ID: 2018115809 (undergraduate)
# NAME: Dohun Kim
# File name: hw5-2.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): os, sys, numpy=1.19.2, matplotlib=3.3.4, scikit-learn=0.24.1

'''
hw5-2.py :
    test various learning techniques with scikit-learn digits dataset
'''

############################## import required packages ##############################
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os, sys

from sklearn.datasets import load_digits
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict

sys.path.append(os.pardir)

from common.gradient import numerical_gradient_2d
from common.multi_layer_net import MultiLayerNet
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Momentum, AdaGrad, Adam
from common.trainer import Trainer
from common.util import smooth_curve
from common.util import shuffle_dataset


################################## define functions ##################################

def weakness_of_sgd():
    X = np.arange(-10, 10, 0.5)
    Y = np.arange(-10, 10, 0.5)
    XX, YY = np.meshgrid(X, Y)
    ZZ = (1 / 20) * XX**2 + YY**2

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap='hot')

    # 그림 6-1 f(x, y) = (1/20) * x**2 + y**2 등고선
    plt.contour(XX, YY, ZZ, 100, colors='k')
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.show()


    def function_2(x):
        if x.ndim == 1:
            return np.sum(x**2)
        else:
            return np.sum(x**2, axis=1)
        
    x0 = np.arange(-10, 10, 1)
    x1 = np.arange(-10, 10, 1)
    X, Y = np.meshgrid(x0, x1)
        
    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient_2d(function_2, np.array([(1/(20**0.5))*X, Y]) )
        
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")
    plt.xlim([-10, 10])
    plt.ylim([-5, 5])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()



def optimizer_compare_naive():

    def f(x, y):
        return x**2 / 20.0 + y**2

    def df(x, y):
        return x / 10.0, 2.0*y

    init_pos = (-7.0, 2.0)
    params = {}
    params['x'], params['y'] = init_pos[0], init_pos[1]
    grads = {}
    grads['x'], grads['y'] = 0, 0


    optimizers = OrderedDict()
    optimizers["SGD"] = SGD(lr=0.95)
    optimizers["Momentum"] = Momentum(lr=0.1)
    optimizers["AdaGrad"] = AdaGrad(lr=1.5)
    optimizers["Adam"] = Adam(lr=0.3)

    idx = 1

    for key in optimizers:
        optimizer = optimizers[key]
        x_history = []
        y_history = []
        params['x'], params['y'] = init_pos[0], init_pos[1]
        
        for i in range(30):
            x_history.append(params['x'])
            y_history.append(params['y'])
            
            grads['x'], grads['y'] = df(params['x'], params['y'])
            optimizer.update(params, grads)
        

        x = np.arange(-10, 10, 0.01)
        y = np.arange(-5, 5, 0.01)
        
        X, Y = np.meshgrid(x, y) 
        Z = f(X, Y)
        
        # 외곽선 단순화
        mask = Z > 7
        Z[mask] = 0
        
        # 그래프 그리기
        plt.subplot(2, 2, idx)
        idx += 1
        plt.plot(x_history, y_history, 'o-', color="red")
        plt.contour(X, Y, Z)
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.plot(0, 0, '+')
        #colorbar()
        #spring()
        plt.title(key)
        plt.xlabel("x")
        plt.ylabel("y")
        
    plt.suptitle('optimizer_compare_naive')
    plt.show()



def optimizer_compare_real_data(data_loader):

    # 0. MNIST 데이터 읽기==========
    (x_train, t_train), (x_test, t_test) = data_loader(normalize=True)

    train_size = x_train.shape[0]
    batch_size = 128
    max_iterations = 2000

    input_size  = x_train.shape[1]
    output_size = len(np.unique(t_train, axis=0))

    # 1. 실험용 설정==========
    optimizers = {}
    optimizers['SGD'] = SGD()
    optimizers['Momentum'] = Momentum()
    optimizers['AdaGrad'] = AdaGrad()
    optimizers['Adam'] = Adam()
    #optimizers['RMSprop'] = RMSprop()

    networks = {}
    train_loss = {}
    for key in optimizers.keys():
        networks[key] = MultiLayerNet(input_size=input_size, 
                                      hidden_size_list=[100, 100, 100, 100],
                                      output_size=output_size)
        train_loss[key] = []    


    # 2. 훈련 시작==========
    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        for key in optimizers.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            optimizers[key].update(networks[key].params, grads)
        
            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)
        
        if i % 100 == 0:
            print( "===========" + "iteration:" + str(i) + "===========")
            for key in optimizers.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(key + ":" + str(loss))


    # 3. 그래프 그리기==========
    markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
    x = np.arange(max_iterations)
    for key in optimizers.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 1)
    plt.legend()

    plt.title('optimizer_compare_mnist')
    plt.show()



def weight_init_activation_histogram():
        
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def ReLU(x):
        return np.maximum(0, x)

    input_data = np.random.randn(1000, 100)  # 1000개의 데이터
    node_num = 100  # 각 은닉층의 노드(뉴런) 수
    hidden_layer_size = 5  # 은닉층이 5개
    activations = {}  # 이곳에 활성화 결과를 저장

    x = input_data

    def get_activation(hidden_layer_size, x, w, a_func=sigmoid):
        for i in range(hidden_layer_size):
            if i != 0:
                x = activations[i-1]

            activations[i] = a_func(np.dot(x, w))
        return activations

    # 히스토그램 그리기
    def get_histogram(activations, suptitle):
        for i, a in activations.items():
            plt.subplot(1, len(activations), i+1)
            plt.title(str(i+1) + "-layer")
            if i != 0: plt.yticks([], [])
            # plt.xlim(0.1, 1)
            # plt.ylim(0, 7000)
            plt.hist(a.flatten(), 30, range=(0,1))
        plt.suptitle(suptitle)
        plt.show()


    vals = {'std=1':1, 'std=0.01':0.01,
            'Xavier':np.sqrt(1.0 / node_num),
            'He':np.sqrt(2.0 / node_num)}

    cases = [(sigmoid, 'std=1'), (sigmoid, 'std=0.01'), (sigmoid, 'Xavier'),
             (ReLU, 'std=0.01'), (ReLU, 'Xavier'), (ReLU, 'He')]

    for z, key in cases:
        suptitle = z.__name__ + ' ' + key
        w = np.random.randn(node_num, node_num) * vals[key]
        activations = get_activation(hidden_layer_size, x, w, z)
        get_histogram(activations, suptitle)



def batch_norm_test(data_loader):

    (x_train, t_train), (x_test, t_test) = data_loader(normalize=True)

    # 학습 데이터를 줄임
    x_train = x_train[:1000]
    t_train = t_train[:1000]

    max_epochs = 20
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01

    input_size  = x_train.shape[1]
    output_size = len(np.unique(t_train, axis=0))

    def __train(weight_init_std):
        bn_network = MultiLayerNetExtend(input_size=input_size, 
                                         hidden_size_list=[100, 100, 100, 100, 100], 
                                         output_size=output_size, 
                                         weight_init_std=weight_init_std, 
                                         use_batchnorm=True)
        network = MultiLayerNetExtend(input_size=input_size, 
                                      hidden_size_list=[100, 100, 100, 100, 100], 
                                      output_size=output_size,
                                      weight_init_std=weight_init_std)
        optimizer = SGD(lr=learning_rate)
        
        train_acc_list = []
        bn_train_acc_list = []
        
        iter_per_epoch = max(train_size / batch_size, 1)
        epoch_cnt = 0
        
        for i in range(1000000000):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
        
            for _network in (bn_network, network):
                grads = _network.gradient(x_batch, t_batch)
                optimizer.update(_network.params, grads)
        
            if i % iter_per_epoch == 0:
                train_acc = network.accuracy(x_train, t_train)
                bn_train_acc = bn_network.accuracy(x_train, t_train)
                train_acc_list.append(train_acc)
                bn_train_acc_list.append(bn_train_acc)
        
                #print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
        
                epoch_cnt += 1
                if epoch_cnt >= max_epochs:
                    break
                    
        return train_acc_list, bn_train_acc_list


    # 그래프 그리기==========
    weight_scale_list = np.logspace(0, -4, num=16)
    x = np.arange(max_epochs)

    for i, w in enumerate(weight_scale_list):
        print( "============== " + str(i+1) + "/16" + " ==============")
        train_acc_list, bn_train_acc_list = __train(w)
        
        plt.subplot(4,4,i+1)
        plt.title("W:" + str(w))
        if i == 15:
            plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
            plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
        else:
            plt.plot(x, bn_train_acc_list, markevery=2)
            plt.plot(x, train_acc_list, linestyle="--", markevery=2)

        plt.ylim(0, 1.0)
        if i % 4:
            plt.yticks([])
        else:
            plt.ylabel("accuracy")
        if i < 12:
            plt.xticks([])
        else:
            plt.xlabel("epochs")
    
    plt.legend(loc='lower right')
    plt.suptitle('batch_norm_test')
    plt.show()



def overfit_weight_decay(data_loader):
    (x_train, t_train), (x_test, t_test) = data_loader(normalize=True)

    # 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
    x_train = x_train[:300]
    t_train = t_train[:300]

    input_size  = x_train.shape[1]
    output_size = len(np.unique(t_train, axis=0))

    # weight decay（가중치 감쇠） 설정 =======================
    val = {False:0, True:0.1}
    suptitle = {False:'Without weight decay', True:'With weight decay'}

    for use_decay in (False, True):
        weight_decay_lambda = val[use_decay]

        network = MultiLayerNet(input_size=input_size, 
                                hidden_size_list=[100, 100, 100, 100, 100, 100], 
                                output_size=output_size,
                                weight_decay_lambda=weight_decay_lambda)
        optimizer = SGD(lr=0.01) # 학습률이 0.01인 SGD로 매개변수 갱신

        max_epochs = 201
        train_size = x_train.shape[0]
        batch_size = 100

        train_acc_list = []
        test_acc_list = []

        iter_per_epoch = max(train_size / batch_size, 1)
        epoch_cnt = 0

        for i in range(1000000000):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            grads = network.gradient(x_batch, t_batch)
            optimizer.update(network.params, grads)

            if i % iter_per_epoch == 0:
                train_acc = network.accuracy(x_train, t_train)
                test_acc = network.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

                if epoch_cnt % 10 == 0:
                    print("epoch:%d, train acc:%.2f%%, test acc:%.2f%%" %(epoch_cnt,
                                                                        train_acc*100,
                                                                        test_acc*100))
                epoch_cnt += 1
                if epoch_cnt >= max_epochs:
                    break
        
        # 그래프 그리기==========
        x = np.arange(max_epochs)
        plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
        plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')

        plt.suptitle(suptitle[use_decay])
        plt.show()



def overfit_dropout(data_loader):
    (x_train, t_train), (x_test, t_test) = data_loader(normalize=True)

    # 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
    x_train = x_train[:300]
    t_train = t_train[:300]

    input_size  = x_train.shape[1]
    output_size = len(np.unique(t_train, axis=0))

    # 드롭아웃 사용 유무와 비울 설정 ========================
    use_dropout = True  # 드롭아웃을 쓰지 않을 때는 False
    dropout_ratio = 0.2
    # ====================================================

    network = MultiLayerNetExtend(input_size=input_size, 
                                  hidden_size_list=[100, 100, 100, 100, 100, 100],
                                  output_size=output_size, 
                                  use_dropout=use_dropout, 
                                  dropout_ration=dropout_ratio)
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                    epochs=301, mini_batch_size=100,
                    optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=False)
    
    print('overfit_dropout(): training...  ')
    trainer.train()

    train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

    # 그래프 그리기==========
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
    plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    
    plt.title('with dropout')
    plt.show()



def hyperparameter_optimization(data_loader):
    (x_train, t_train), (x_test, t_test) = data_loader(normalize=True)

    # 결과를 빠르게 얻기 위해 훈련 데이터를 줄임
    x_train = x_train[:500]
    t_train = t_train[:500]
    
    input_size  = x_train.shape[1]
    output_size = len(np.unique(t_train, axis=0))

    # 20%를 검증 데이터로 분할
    validation_rate = 0.20
    validation_num = int(x_train.shape[0] * validation_rate)
    x_train, t_train = shuffle_dataset(x_train, t_train)
    x_val = x_train[:validation_num]
    t_val = t_train[:validation_num]
    x_train = x_train[validation_num:]
    t_train = t_train[validation_num:]


    def __train(lr, weight_decay, epocs=50):
        network = MultiLayerNet(input_size=input_size, 
                                hidden_size_list=[100, 100, 100, 100, 100, 100],
                                output_size=output_size, 
                                weight_decay_lambda=weight_decay)
        trainer = Trainer(network, x_train, t_train, x_val, t_val,
                        epochs=epocs, mini_batch_size=100,
                        optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
        trainer.train()

        return trainer.test_acc_list, trainer.train_acc_list


    # 하이퍼파라미터 무작위 탐색======================================
    optimization_trial = 100
    results_val = {}
    results_train = {}
    for _ in range(optimization_trial):
        # 탐색한 하이퍼파라미터의 범위 지정===============
        weight_decay = 10 ** np.random.uniform(-8, -4)
        lr = 10 ** np.random.uniform(-6, -2)
        # ================================================

        val_acc_list, train_acc_list = __train(lr, weight_decay)
        print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
        key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
        results_val[key] = val_acc_list
        results_train[key] = train_acc_list

    # 그래프 그리기========================================================
    print("=========== Hyper-Parameter Optimization Result ===========")
    graph_draw_num = 20
    col_num = 5
    row_num = int(np.ceil(graph_draw_num / col_num))
    i = 0

    for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
        print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

        plt.subplot(row_num, col_num, i+1)
        plt.title("Best-" + str(i+1))
        plt.ylim(0.0, 1.0)
        if i % 5: plt.yticks([])
        plt.xticks([])
        x = np.arange(len(val_acc_list))
        plt.plot(x, val_acc_list)
        plt.plot(x, results_train[key], "--")
        i += 1

        if i >= graph_draw_num:
            break
    
    plt.suptitle('Hyper-Parameter Optimization Result')
    plt.show()



############################# define data loader function ############################

def sklearn_load_digits(normalize=True):
    
    # load data from gzip files
    x, t = load_digits(return_X_y=True)

    # normalization of input data
    if normalize:
        x = x / x.max()

    # shuffle data randomly
    x, t = shuffle_dataset(x, t)

    # split dataset into training(80%) and test(20%) sets
    train_rate = 0.8
    labels = np.unique(t, axis=0).tolist()

    train_idx, test_idx = [], []

    for label in labels:
        all_idx = list(np.where(t == label)[0])
        num_train = int(len(all_idx) * train_rate)

        train_idx += all_idx[:num_train]
        test_idx  += all_idx[num_train:]

    x_train, t_train = x[train_idx], t[train_idx]
    x_test,  t_test  = x[test_idx],  t[test_idx]

    x_train, t_train = shuffle_dataset(x_train, t_train)

    return (x_train, t_train), (x_test, t_test)


######################################## main ########################################

if __name__ == '__main__':

    funcs = [
        (weakness_of_sgd, False),
        (optimizer_compare_naive, False),
        (optimizer_compare_real_data, True),
        (weight_init_activation_histogram, False),
        (batch_norm_test, True),
        (overfit_weight_decay, True),
        (overfit_dropout, True),
        (hyperparameter_optimization, True)
    ]

    data_loader = sklearn_load_digits

    for f, use_data_loader in funcs:
        print('running ' + f.__name__ + '() ...')
        if use_data_loader:
            f(data_loader)
        else:
            f()
        print()
