from keras.datasets import mnist,cifar10,cifar100
import numpy as np
import os

def get_mnist():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train/255.0      # 这里是专门针对图像数据的归一化
    x_test = x_test/255.0
    x_train = np.reshape(x_train,(-1,28,28,1))
    x_test = np.reshape(x_test,(-1,28,28,1))

    return x_train,y_train,x_test,y_test     # x_train (60000, 28, 28, 1)  x_test (10000, 238, 28, 1)

def get_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def get_imagenet():
    x_train = np.load('G:/datasets_and_networks/imagenet/data/x_train.npy')
    x_test = np.load('G:/datasets_and_networks/imagenet/data/x_test.npy')
    y_train = np.load('G:/datasets_and_networks/imagenet/data/y_train.npy')
    y_test = np.load('G:/datasets_and_networks/imagenet/data/y_test.npy')

    return x_train, y_train, x_test, y_test

def load_drebin_data():
    x_train = np.load(os.path.join('./', 'datasets/drebin/train/mama_family_ori_train_data.npy'))
    y_train = np.load(os.path.join('./', 'datasets/drebin/train/mama_family_ori_train_label.npy'))
    x_test = np.load(os.path.join('./', 'datasets/drebin/test/mama_family_testori_data.npy'))
    y_test = np.load(os.path.join('./', 'datasets/drebin/test/mama_family_testori_label.npy'))

    x_train = np.reshape(x_train, (-1, 11, 11, 1))
    x_test = np.reshape(x_test, (-1, 11, 11, 1))

    return (x_train, y_train), (x_test, y_test)
