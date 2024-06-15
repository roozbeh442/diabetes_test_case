import sklearn as sk
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import tensorflow as tf

def generate_data():
    '''loads the diabetes dataset in Normalized format (zero mean and unit std). 
       uses the last 20 data points as test dataset. no inputs required.'''
    X, y = load_diabetes(return_X_y=True, scaled=True)
    y = (y - y.mean())/y.std()
    X_train = X[:442-20]
    y_train = y[:442-20]
    X_test  = X[-20:]
    y_test  = y[-20:]
    return X_train, y_train, X_test, y_test

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