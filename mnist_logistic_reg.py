# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 18:00:36 2020

This script will classify MNSIT dataset

takes the path to the train and test data

fits with Logistic Regression

@author: Elijah
"""


# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

import multiprocessing
from joblib import Parallel, delayed
#%% functions
def dim_reduction(data,n_components):

    # INPUTS
    # % data : df for PCA
    # % n_components : threshold for explained variance, 0 < n_components < 1

    # mean subtract
    data = data.sub(data.mean(axis=1), axis=0)
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    
    # pca
    pca = PCA(n_components=n_components,whiten=False, random_state=10)
    pca.fit(data)

    # pick number of components if 90% is not reached
    cum_explained = np.cumsum(pca.explained_variance_ratio_)*100
    sns.lineplot(np.arange(1,pca.n_components_+1), cum_explained)
    plt.show()
    
    return pca.transform(data), pca

def model_selection(X,y,C_vals):
    # selects the model with the best regularization paramaters
    
    mean_accuracies = []
    for c in C_vals:
        
        #accuracies = cross_val_score(SVC(kernel='rbf', gamma=0.001, C=c), X, y, cv=4, n_jobs=4) # k=8 crossvalidation
        accuracies = cross_val_score(LogisticRegression(C=c, penalty='l1',tol=0.1), X, y, cv=4, n_jobs=4) # k=8 crossvalidation
        
        mean_accuracies.append(accuracies.mean())
    
    return mean_accuracies

#%% Run the code for preprocessing and model selection
if __name__ == '__main__':
    
    #%% load data
    train = pd.read_csv('C:/Users/eap40/Desktop/kaggle/mnist/train.csv')
    y = train['label']
    train = train.iloc[:,1:]
    
    test = pd.read_csv('C:/Users/eap40/Desktop/kaggle/mnist/test.csv')
    test = test.sub(test.mean(axis=1), axis=0)
    
    #%% Dimensionality reduction
    n_components = 0.9
    X, pca = dim_reduction(train, n_components) # fit only to training data set

    X_test = pca.transform(test.values)
    
    #%% Select model
    t0 = time.time()
    
    C_vals = np.logspace(-4, 4, 9) #logspace

    mean_accuracies = model_selection(X,y,C_vals)
    
    C = C_vals[np.argmax(mean_accuracies)]
    clf = LogisticRegression(C=C, penalty='l1',tol=0.1)
    clf.fit(X,y)

    # model predictions on test data 
    predictions = clf.predict(X_test)

    # save data
    output = pd.DataFrame(np.arange(1, len(predictions)+1), columns={'ImageId'})
    output['Label'] = predictions
    output.to_csv('C:\Users\eap40\Desktop\kaggle\mnist\submission.csv', index=False)
    t1 = time.time()
    total_time = t1 - t0
    
    
    #%% plot results
    plt.figure()
    sns.lineplot(C_vals, mean_accuracies, linewidth=2)
    plt.xlabel('C values')
    plt.ylabel('Mean accuracy')
    sns.despine()
