import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Highway
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback
import keras.backend as K
import pdb
import time
def get_data(config):
    if config['data'] == 'syn8':
        config['input_size'] = (50,)
        config['output_size'] = 2
    elif config['data'] == 'higgs':
        config['input_size'] = (28,)
        config['output_size'] = 2
    elif config['data'] == 'susy':
        config['input_size'] = (18,)
        config['output_size'] = 2
    elif config['data'] == 'cd6' or config['data'] == 'cd7':
        config['input_size'] = (50,)
    elif config['data'] == 'cd3' or config['data'] == 'cd4':
        config['input_size'] = (25,)
    config['output_size'] = 2
    return config

def build_model(config):
    config = get_data(config)

    base_name = 'out'
    if config['hedge'] == True:
        outs = ['']*config['n_layers']
        out_name = ['']*config['n_layers']
        N = config['n_layers']
        for i in range(len(outs)):
            outs[i] = base_name + str(i)
            out_name[i] = base_name + str(i)
    else:
        outs = base_name
        out_name = [base_name]
        N = config['n_layers'] - 1
    in_name = 'in0'

    inputs = Input(config['input_size'], name = in_name)
    
    for j in range(N):
        if j == 0:
            layer = Dense(config['hidden_num'])(inputs)
            layer = Activation(config['activation'])(layer)

            if config['hedge'] == True:
                outs[j] = Dense(config['output_size'], activation = 'softmax', name = outs[j])(layer)
            continue
        if config['Highway'] == False:
            layer = Dense(config['hidden_num'])(layer)
            layer = Activation(config['activation'])(layer)
        else:
            layer = Highway(activation = config['activation'])(layer)
            
        if config['hedge'] == True:
            outs[j] = Dense(config['output_size'], activation = 'softmax', name = outs[j])(layer)
    if config['hedge'] == False:
        outs = Dense(config['output_size'], activation = 'softmax', name = outs)(layer)
    model = Model(input = inputs , output = outs)

    return (model, in_name, out_name)

# add self.masks, self.weighted_losses
class MyCallback(Callback):
    def __init__(self,w,  beta = 0.9,  names = [], hedge = False, log_name = None):
        self.weights = w
        self.beta = beta
        self.names = names
        self.l = []
        self.hedge = hedge
        self.acc = []
        self.log_name = log_name
    def on_batch_end(self, batch, logs = {}):
        self.l.append(logs.get('loss'))
        losses = [logs[name] for name in self.names]
        self.acc.append(logs.get('acc'))
        if self.hedge:

            M = sum(losses)
            losses = [loss / M for loss in losses]
            min_loss = np.amin(losses)
            max_loss = np.amax(losses)
            range_of_loss = max_loss - min_loss
            losses = [(loss-min_loss)/range_of_loss for loss in losses]

            alpha = [self.beta ** loss for loss in losses]
            
            try:
                alpha = [a * w for a, w in zip(alpha, self.weights)]
            except ValueError:
                pass

            alpha = [ max(0., a) for a in alpha]
            M = sum(alpha)
            alpha = [a / M for a in alpha]
            
            self.weights = alpha 
    def on_batch_begin(self, epoch, logs={}):
        self.model.holder = (self.weights)
        #print(self.model.holder)
