import os, sys, getopt 
import yaml
import cPickle

import numpy as np

import keras
import keras.callbacks
from keras.datasets import mnist
from keras.utils.visualize_util import plot
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam, RMSprop
from model import build_model, MyCallback
from keras.callbacks import CSVLogger
from data import load
import pdb

def build_data_dict(in_name, out_name, in_data, out_data):
    in_dict = dict()
    in_dict[in_name] = in_data
    
    out_dict = dict((k, out_data) for k in out_name)
    return (in_dict, out_dict)

def build_loss_weight(config):
    if config['hedge'] == False:
        w = [1.]
    elif config['loss_weight'] == 'ave':
        w = [1./ config['n_layers']]* config['n_layers']
    return w
def main(arg, idx=0):
    config = {'learning_rate': 1e-3,
              'optim': 'Adam',
              'batch_size': 1,
              'nb_epoch': 50,
              'n_layers': 3,
              'hidden_num': 100,
              'activation': 'relu',
              'loss_weight': 'ave',
              'adaptive_weight': False,
              'data': 'mnist',
              'hedge': False,
              'Highway': False,
              'momentum': 0.,
              'nesterov': False,
              'log': 'mnist_hedge.log'}

    configfile = ''
    helpstring = 'main.py -c <config YAML file>'
    try:
        opts, args = getopt.getopt(arg, "hc:", ["config"])
    except getopt.GetoptError:
        print(helpstring)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print (helpstring)
            yamlstring = yaml.dump(config,default_flow_style=False,explicit_start=True)
            print("YAML configuration file format:")
            print("")
            print("%YAML 1.2")
            print(yamlstring)
            sys.exit()

        elif opt in ('-c', '--config'):
            configfile = arg

        print("Config file is %s" % configfile)

    if os.path.exists(configfile):
        f = open(configfile)
        user_config = yaml.load(f.read())
        config.update(user_config)
    
    print("Printing configuration:")
    for key,value in config.iteritems():
        print("  ",key,": ",value)

    (X_train, Y_train, X_test, Y_test, nb_classes) = load(config['data'])
    '''
    X_train = X_train[:1000]
    Y_train = Y_train[:1000]
    X_test = X_test[:1000]
    Y_test = Y_test[:1000]
    '''
    model, in_name, out_name = build_model(config)
    if len(out_name) == 1:
        out_name_loss = ['loss']
    else:
        out_name_loss = [s + '_loss' for s in out_name]

    model.summary()
    
    #plot(model, to_file = 'model.png')
    
    optim = eval(config['optim'])(lr = config['learning_rate'], momentum = config['momentum'], nesterov = config['nesterov'])
    in_dict, out_dict = build_data_dict(in_name, out_name, X_train, Y_train)
    in_val, out_val = build_data_dict(in_name, out_name, X_test, Y_test)
    loss_dict = dict((k, 'categorical_crossentropy') for k in out_name) 
  
    loss_weights = build_loss_weight(config)
    my_callback = MyCallback(loss_weights, names = out_name_loss, hedge = config['hedge'], log_name = config['log'])
    #csv  = CSVLogger(config['log'])
    model.compile(optimizer = optim, loss = loss_dict, hedge = config['hedge'],loss_weights = loss_weights, metrics = ['accuracy'])
    model.fit(in_dict, out_dict, nb_epoch = config['nb_epoch'], batch_size = config['batch_size'], callbacks=[my_callback])
    cumLoss = np.cumsum(my_callback.acc)
    indexOfLoss = np.arange(len(cumLoss))+1
    cumAverageLoss = cumLoss/indexOfLoss
    filename = (config['log'] + '_' + str(idx) + '.acc')
    np.savetxt(filename, cumAverageLoss, delimiter=',')
    
    return my_callback
if __name__ == '__main__':
    #for i in range(5):
    my_callback = main(sys.argv[1:], 0)
