# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:01:38 2017

@author: Admin
"""

#import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense
import keras.backend as K


def simple_nn():
    # the four different states of the XOR gate
    x_train= [[0,0],[0,1],[1,0],[1,1]]
    # the four expected results in the same order
    y_train = [[0],[1],[1],[0]]
    
    model = Sequential()
    keras.initializers.Zeros()
    
    model.add(Dense(128, input_dim=2, activation='relu'))
    model.add(Dense(32, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='relu'))
    
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', precision ])
    
    model.fit(x_train, y_train, nb_epoch=1000, verbose=2)
    print (model.predict(x_train))
    return model


def performance():
   
    X_test=[[0,1],[1,1]]
   # y_test=[[1],[0]]
    y_test=[[1],[1]]
    
    model=simple_nn()
    
    y_pred=model.predict(X_test)
    print (y_pred)
    
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test Accuracy:', score[1])
    print('Test F1-measure:', score[2])
    print('Test Precision:', score[3])
    print('Test Recall:', score[4])
    

def metric(y_test, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_test, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    return c1,c2,c3
    
def precision(y_test, y_pred):
    m=metric(y_test,y_pred)
    # How many selected items are relevant?
    precision = m[0] / m[1]
    return precision

def recall(y_test, y_pred):
    m=metric(y_test,y_pred)
    # How many relevant items are selected?
    recall = m[0] / m[2]
    return recall

def f1_score(y_test, y_pred):
    p=precision(y_test, y_pred)
    r=recall(y_test, y_pred)
    # Calculate f1_score
    f1_score = 2 * (p * r) / (p + r)
    return f1_score

simple_nn()