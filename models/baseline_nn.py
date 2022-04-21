import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np


class baselineNN():

    def __init__(self,config):
        self.config = config

    def train(self,x_train,y_train,x_valid,y_valid):

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)

        x_train = (x_train-x_mean)/x_std
        x_valid = (x_valid-x_mean)/x_std

        
        self.model = models.Sequential()

        self.model.add(layers.Dense(self.config['hidden_layers_nodes'], activation=self.config['activation'], input_shape=[x_train.shape[1]]))
        self.model.add(layers.Dense(self.config['hidden_layers_nodes'], activation=self.config['activation']))
        self.model.add(layers.Dense(1))

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config['patience'],restore_best_weights=True)

        self.model.compile(optimizer=self.config['optimizer'], loss='mse', metrics=['mse'])
        history = self.model.fit(x_train, y_train, batch_size=self.config['batch_size'], validation_data=(x_valid, y_valid), \
            epochs=1000,callbacks=[callback],verbose=0)
        
        return x_mean, x_std, min(history.history['val_loss'])
        
        



