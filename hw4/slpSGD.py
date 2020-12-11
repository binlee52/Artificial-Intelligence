import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(df):
    """
        X (ndarray): include the columns of the features and bias, shape == (N, D+1)
        y (ndarray): label vector, shape == (N, )
    """
    
    N = df.shape[0]         # the number of data samples
    
    X = np.hstack([np.ones((N, 1)), df.values[:, 1:]])
    y = df.values[:, 0]

    return (X, y)

def sigmoid(z):
    """Sigmoid function""" 
    ## Fill In Your Code Here ##
    
    # compute sigmoid function 
    o = 1/(1+np.exp(-z))
    ############################
    return o

class SLPerceptron():
    
    def __init__(self, rate=0.1, iterations=100000):
        self.rate = rate
        self.iterations = iterations
            
    def fit(self, X, y):
        """Perform Single-Layer Perceptron training using SGD"""
        N = X.shape[0]         # the number of data samples
        D = X.shape[1]         # the number of features (x0, x1, ..., xD), including bias x0 = 1
        w = np.zeros(D)      # (w0, w1, ..., wD)
        
        total_loss = 0
        self.losses = []
        
        for i in range(self.iterations):
            
            xi = X[i%N]      # i th input x = (x0, x1, ..., xD)
            yi = y[i%N]      # i th output y
            
            ## Fill In Your Code Here ##
            ## (use class methods and variables 
            ##  ex> self.activation, self.get_loss, self.rate)
            # compute output of neuron (forward computation)
            oi = self.activation(xi, w)
            # compute delta
            delta = (yi - oi) * oi * (1 - oi)
            # update w
            w = w + self.rate * delta * xi
            # compute loss and accumulate to total loss
            total_loss = self.get_loss(yi, oi)
            ############################

            # each epoch
            if (i+1)%N == 0 :
                loss = total_loss / N               # loss = mean squared error
                self.losses.append(loss)
                total_loss = 0
    
        return w
    
    def activation(self, x, w):
        """Output of neuron for x with w (sigmoid(xw))"""
        ## Fill In Your Code Here ##
        z = x@w
        o = sigmoid(z)
        # compute z = xw
        # compute output
        
        ############################
        return o

    def get_loss(self, yi, oi):
        """Calculate squared error"""

        loss = (1/2)*(yi - oi)**2
        
        return loss
        
    def get_accuracy(self, X, y, w):
        """Return accuracy for given X, y and w.
           Accuracy is the fraction of right predictions.
        """
        y_hat = self.predict(X, w)
        
        diff = np.abs(y - y_hat)
        num_incorrect = diff.sum()
        N = X.shape[0]
        
        accuracy = 1 - (num_incorrect / N)

        return accuracy
            
    def predict(self, X, w):
        """Predict labels(0 or 1) for X with parameters w."""
        ## Fill In Your Code Here ##

        # compute output
        o = self.activation(X, w)
        # predict label y_hat
        y_hat = np.where(o >= 0.5, 1, 0)
        ############################
        return y_hat
