import keras
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout
from keras.callbacks import Callback

def make_model():
    inputs = Input((100,))
    x1 = Dense(100, activation='relu')(inputs)
    x2 = Dense(100, activation='relu')(x1)
    x3 = Dense(100, activation='relu')(x2)

    out1 = Dense(10, activation = 'softmax', name = 'out1')(x1)
    out2 = Dense(10, activation = 'softmax', name = 'out2')(x2)
    out3 = Dense(10, activation = 'softmax', name = 'out3')(x3)

    model = Model(input = inputs, output = [out1,out2,out3])
    return model

class MyCallback(Callback):
    def __init__(self,w,  beta = 0.99,  names = [], hedge = False, log_name = 'exp'):
        self.weights = w
        self.beta = beta
        self.l = []
        self.hedge = hedge
        self.names = names
        self.acc = []
    #def on_train_begin(self, logs={}):
        
    def on_batch_end(self, batch, logs = {}):
        self.acc.append(logs.get('weighted_acc'))
        losses = [logs[name] for name in self.names]

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

        alpha = [ max(0.01, a) for a in alpha]
        M = sum(alpha)
        alpha = [a / M for a in alpha]
        self.weights = alpha

    def on_batch_begin(self, epoch, logs={}):
        self.model.holder = (self.weights)
    def on_train_end(self, epoch, logs={}):
        self.model.holder = (self.weights)
if __name__ == '__main__':
    x = np.random.randn(1000, 100)
    y = np.random.randn(1000, 10)
    Y = {'out1':y, 'out2':y, 'out3':y}
    model = make_model()
    optim = SGD()
    out_name_loss = [s + '_loss' for s in ['out1','out2','out3']]   
    my_callback = MyCallback([1.0/3]*3, names = out_name_loss, hedge = True)
    model.compile(optimizer=optim, loss = ['mse','mse','mse'], hedge = True, loss_weights = [1.0/3]*3, metrics =['accuracy'])
    model.fit(x,Y , nb_epoch= 1, batch_size=50, callbacks=[my_callback])

