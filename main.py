import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 

data = pd.read_csv('/Users/antonyshibupaul/Documents/VScode/digit-recognizer/train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev=data[0:1000].T
Y_dev= data_dev[0]
X_dev=data_dev[1:n]
X_dev = X_dev / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_,m_train = X_train.shape

def init_params():
    # Previous: W1 = np.random.rand(10, 784)-0.5
    W1 = np.random.rand(64, 784)-0.5 # Change 10 to 64
    b1 = np.random.rand(64,1)-0.5   # Change 10 to 64
    # Previous: W2 = np.random.rand(10, 10)-0.5
    W2 = np.random.rand(10, 64)-0.5 # Change second 10 to 64
    b2 = np.random.rand(10,1)-0.5
    return W1, b1, W2, b2

#Relu function
def ReLu(Z):
    return np.maximum(0,Z)
    
#softmax fucnction
def softmax(Z):
    A = np.exp(Z)/sum(np.exp(Z))
    return A
    
#forward propogation
def forward_propogation(W1,b1,W2,b2,X):
    Z1= W1.dot(X) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
    
#one hot encoding
def ohe(Y):
    num_classes = Y.max() + 1
    ohe_Y = np.zeros((Y.size, num_classes))
    ohe_Y[np.arange(Y.size), Y] = 1
    ohe_Y = ohe_Y.T
    return ohe_Y

#ReLu derivative function
def deriv_ReLu(Z):
    return Z>0 

#backward propogation
def backward_propogation(Z1,A1,Z2,A2,W1,W2,X,Y):
    m=Y.size
    ohe_Y = ohe(Y)
    dZ2 = A2 - ohe_Y
    dW2 = 1/m*dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2)
    dW1 = 1/m*dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1,b1,W2,b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
    
def gradient_descent(X,Y,iterations,alpha):
    W1,b1,W2,b2 = init_params()
    for i in range(iterations):
        Z1,A1,Z2,A2 = forward_propogation(W1,b1,W2,b2,X)
        dW1,db1,dW2,db2 = backward_propogation(Z1,A1,Z2,A2,W1,W2,X,Y)
        W1,b1,W2,b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if(i%50==0):
            print("Iteration:",i)
            print("Accuracy",get_accuracy(get_predictions(A2),Y))
    return W1,b1,W2,b2

W1,b1,W2,b2 = gradient_descent(X_train, Y_train,500,0.10)
