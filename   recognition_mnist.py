#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:50:02 2019

@author: shuma
"""

#import modules
import keras
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(x_test.shape)
print(y_train)

#変数の定義
x_train = x_train.reshape(60000,784)/255
x_test = x_test.reshape(10000,784)/255

bias0 = [0]*15
bias1 = [0]*10
re_bias0 = [0]*15
re_bias1 = [0]*10
weight0 = [[0 for i in range(784)]for i in range(15)]
weight1 = [[0 for i in range(15)]for i in range(10)]
re_weight0 = [[0 for i in range(784)]for i in range(15)]
re_weight1 = [[0 for i in range(15)]for i in range(10)]

In0 = [0]*15
In1 = [0]*10
Out0 = [0]*15
Out1 = [0]*10

In0_test = [0]*15
In1_test = [0]*10
Out0_test = [0]*15
Out1_test = [0]*10


#関数の定義
def sigmoid(x):
    return 1/( 1 + np.exp(-x) )

def sub_sigmoid(x):
    return  (1-sigmoid(x))*sigmoid(x) 

def softmax(x_array):
    a = np.max(x_array)
    exp_x = np.exp(x_array - a)
    sum_exp_x = np.sum(exp_x)
    y_array = exp_x/sum_exp_x
    return y_array

def delta(num,t_n,Op1,Ip1,we1):
    sum_1 = 0
    for i in range(10):
        sum_1 += (Op1[i]-t_n[i])*we1[i][num]*sub_sigmoid(Ip1[i])
    return sum_1

def back_propagation(Out0,Out1,In0,In1,t_num,x_t):
    
    global weight0
    global weight1
    global bias0
    global bias1
    
    for i in range(10):
        for j in range(15):
            re_weight1[i][j] = (Out1[i]-t_num[i])*sub_sigmoid(In1[i])
            weight1[i][j] -= learning_rate*re_weight1[i][j]*Out0[j]
    
    for i in range(15):
        for j in range(784):
            re_weight0[i][j] = delta(i,t_num,Out1,In1,weight1)*sub_sigmoid(In0[i])
            weight0[i][j] -= learning_rate*re_weight0[i][j]*x_t[j]
    
    for i in range(10):
        re_bias1[i] = (Out1[i]-t_num[i])*sub_sigmoid(In1[i])
        bias1[i] -= learning_rate*re_bias1[i]
        
    for i in range(15):
        re_bias0[i] = delta(i,t_num,Out1,In1,weight1)*sub_sigmoid(In0[i])
        bias0[i] -= learning_rate*re_bias0[i]
        
def accuracy()(y_list,t_list):
    max_y = np.argmax(y_list,axis=1)
    max_t = np.argmax(t_list,axis=1)
    return np.sum(max_y == max_t)/100

def sum_of_squares_error(y,t):
    return 0.5*np.sum((y-t)**2)

for i in range(15):
    bias0[i] = np.random.rand()*0.1
for i in range(10):
    bias1[i] = np.random.rand()*0.1
for i in range(15):
    for j in range(784):
        weight0[i][j] = np.random.randn()*0.1
for i in range(10):
    for j in range(15):
        weight1[i][j] = np.random.randn()*0.1

        
learning_rate = 0.1
epoch = 3

accuracy_range = []
sum_of_squares_error_range = []

for l in range(epoch):
    for k in range(600):   
        train_prediction = []
        train_answer = []
        print(str(l*600+k)+"順目の学習中")
        for j in range(100):
            for i in range(15):
                In0[i] = np.dot(x_train[k*100+j],weight0[i])+bias0[i]
                Out0[i] = sigmoid(In0[i])
            for i in range(10):
                In1[i] = np.dot(Out0,weight1[i])+bias1[i]
            
            Out1 = softmax(In1)        
            true_num = [0]*10
            true_num[y_train[k*100+j]] = true_num[y_train[k*100+j]]+1
        
            train_prediction.append(Out1)
            train_answer.append(true_num)

            back_propagation(Out0,Out1,In0,In1,true_num,x_train[k*100+j])
            
        accuracy_range.append(accuracy(train_prediction,train_answer))
        sum_of_squares_error_range.append(sum_of_squares_error(Out1,true_num))
        
        print("accuracy= "+str(accuracy(train_prediction,train_answer)))
        print("sum_of_squares_error = "+str(sum_of_squares_error(Out1,true_num)))
        

x = [0]*600*epoch
for i in range(600*epoch):
    x[i] = i
y = accuracy_range
z = sum_of_squares_error_range

plt.plot(x,y,label="accuracy" )
plt.plot(x,z,label="sum_of_squares_error")
plt.legend(loc = "lower right")
#plt.savefig(your_path)
