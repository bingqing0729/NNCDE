import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime

tf.compat.v1.enable_eager_execution()

class custom_loss(keras.losses.Loss):
    def __init__(self,batch_size=100,name="custom_loss"):
        super().__init__(name=name)
        self.batch_size = batch_size
    
    def call(self, y_true, y_pred):
        return tf.divide(-tf.reduce_sum(tf.multiply(y_pred,tf.expand_dims(y_true,-1))),self.batch_size) # loss component 1

class hazardNN:
    def __init__(self,config={}):
        self.config = config

    def train(self,x_train,fail_train,x_valid,fail_valid):
        
        
        x_train = tf.ragged.constant(x_train).to_tensor()
        fail_train = tf.ragged.constant(fail_train).to_tensor()
        x_valid = tf.ragged.constant(x_valid).to_tensor()
        fail_valid = tf.ragged.constant(fail_valid).to_tensor()

        self.input_dim = x_train.shape[2]

        dt_train = tf.expand_dims(x_train[:,:,self.input_dim-1]-x_train[:,:,self.input_dim-2],-1)
        dt_valid = tf.expand_dims(x_valid[:,:,self.input_dim-1]-x_valid[:,:,self.input_dim-2],-1)
        

        x_mean = tf.math.reduce_mean(x_train,axis=[0,1])
        x_std = tf.math.reduce_std(x_train,axis=[0,1])
        x_train = (x_train-x_mean)/x_std
        x_valid = (x_valid-x_mean)/x_std
        

        inputs = keras.Input(shape=(None,self.input_dim), name="x",dtype=np.float32)
        dt = keras.Input(shape=(None,1),name="dt",dtype=np.float32)

        x1 = layers.Dense(self.config['hidden_layers_nodes'], activation=self.config['activation'])(inputs)
        #x1b = layers.BatchNormalization()(x1)
        #x1d = layers.Dropout(0.2)(x1b)
        x2 = layers.Dense(self.config['hidden_layers_nodes'], activation=self.config['activation'])(x1)
        #x2b = layers.BatchNormalization()(x2)
        #x2d = layers.Dropout(0.2)(x2b)
        outputs = layers.Dense(1, name="log_hazard")(x2)
        self.model = keras.Model(inputs=[inputs,dt], outputs=outputs)

        l = tf.divide(tf.reduce_sum(tf.multiply(tf.exp(outputs),dt)),self.config['batch_size']) # loss component 2
        self.model.add_loss(l)

        if self.config['optimizer']=='adam':
            optimizer=keras.optimizers.Adam(lr=self.config['learning_rate'])
        elif self.config['optimizer']=='sgd':
            optimizer=keras.optimizers.SGD(lr=self.config['learning_rate'])

        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config['patience'],restore_best_weights=True)
        self.model.compile(optimizer=optimizer,loss=custom_loss(self.config['batch_size']))
        history = self.model.fit([x_train,dt_train], fail_train, validation_data=([x_valid,dt_valid],fail_valid),\
            batch_size=self.config['batch_size'], epochs=2000,callbacks=[callback],verbose=0)
        

        return x_mean, x_std, min(history.history['val_loss'])
        


