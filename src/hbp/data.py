import numpy as np
import os 
from scipy.io import loadmat
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K

def load(dataset):
    assert dataset in ['higgs', 'cd1','cd2','cd3','cd4','cd5','cd6','cd7','syn8','susy']
    data = loadmat('../../data/' + dataset + '.mat')
    X_train = data['X_train']
    X_test = data['X_test']
    Y_train = data['Y_train']
    Y_test = data['Y_test']

    nb_classes = 2
    return (X_train, Y_train, X_test, Y_test, nb_classes)


