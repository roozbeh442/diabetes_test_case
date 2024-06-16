import sklearn as sk
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from data_models import features,properties, target, line_input

def simple_line(data:line_input):
    x = data.x
    w = data.w
    b = data.b
    return x * w + b

def generate_data():
    '''loads the diabetes dataset in Normalized format (zero mean and unit std). 
       uses the last 20 data points as test dataset. no inputs required.
       this function also returns pydantic models with properties of data'''
    X, y = load_diabetes(as_frame=True, scaled=False,return_X_y=True)
    X_train = X[:442-20]
    y_train = y[:442-20]
    X_test  = X[-20:]
    y_test  = y[-20:]
    data = features()
    target_prop = target(value=None,
                        max=max(y),
                        min=min(y),
                        mean = y.mean(),
                        std= y.std())
    for item in data.__fields__.items():
        setattr(data,item[0],properties(value=None,
                                        max=max(X[item[0]]),
                                        min=min(X[item[0]]),
                                        mean = X[item[0]].mean(),
                                        std= X[item[0]].std()))
    return X_train, y_train, X_test, y_test, data, target_prop

def create_dataloader(X,y,batch_size):
    '''Create dataloader with defined batch_size from feature and target lists
        X (list of 10 floats): feature list
        y (list of floats): target list
        batch_size (int): user defined batch size '''
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))
    dataloader = dataset.shuffle(buffer_size=len(X)).batch(batch_size)
    return dataloader

class RegressionNetwork():
    def __init__(self, X, y, data_props, target_props,batch_size, epochs):
        self.X = X
        self.y = y
        self.data_props = data_props
        self.target_props = target_props
        self.epochs = epochs
        self.batch_size = batch_size
        '''Build a fully connected network with two hidden layers.'''
        model = models.Sequential()
        model.add(layers.Dense(25, activation='relu', input_shape = (10,))),
        model.add(layers.Dense(25,activation='relu')),
        model.add(layers.Dense(1))
        self.model = model
    def create_dataloader(self):
        '''Create dataloader with defined batch_size from feature and target lists
            X (list of 10 floats): feature list
            y (list of floats): target list
            batch_size (int): user defined batch size '''   
        self.Normalize_data()    
        X_tensor = tf.convert_to_tensor(self.X, dtype=tf.float32)
        y_tensor = self.y #tf.convert_to_tensor(self.y, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))
        self.dataloader = dataset.shuffle(buffer_size=len(self.X)).batch(self.batch_size)
        return self.dataloader
    
    def train_network(self):
        self.model.compile(optimizer = 'adam', loss = 'mse')
        history = self.model.fit(self.dataloader,batch_size = self.batch_size, epochs=self.epochs)
        return history
        

    def Normalize_data(self):
        for cols in  self.X.columns:
            temp_att = getattr(self.data_props,cols)
            self.X[cols] = tf.convert_to_tensor(self.X[cols])
            self.X[cols] = (self.X[cols] - temp_att.mean)/temp_att.std
        self.y = tf.convert_to_tensor(self.y)
        self.y = (self.y - self.target_props.output.mean)/self.target_props.output.std 
           
            
