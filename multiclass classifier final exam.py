# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 17:03:29 2021

@author: louise
"""
#use matmul
#first row is the label
#binary classification represents number 1 or 2
#rest is pixels 784 per picture
#pixel between 0-255, zero white, 255 black
#training 70%
#testing 30%
#w*x w all vector coefficients, x entire row, use mathmul thing
#cant do graph would be 784D
#add part for testing
#confusion matrix and that mess
#accuracy

import numpy as np
import tensorflow as tf
# import tensorflow.math as math
from tensorflow import linalg, math

import random
DTYPE=tf.float32

def get_data():
    """read data, returns data as list 
    with list of rows which contain values as int"""
    inp = open("multiple_digit_dataset.csv", "r", encoding="utf-8")
    data = []
    for line in inp.readlines():
        split_line = [float(x) for x in line.split(",")]
        norm_line = [split_line[0]] +[x/255 for x in split_line[1:]]
        
        data.append(norm_line)
    #print(data)
    #split data in training and testing
    length_data = len(data)
    training_i = int(length_data*0.7)
    random.shuffle(data) #shuffle data

    training_data = data[0:training_i]
    testing_data = data[training_i: -1]
    return training_data, testing_data

def initialize_data(data):
    """initialize data"""   
    Xs = []
    Ys=[]
    for line in data:
        sub = [0,0,0,0]
        Xs.append(line[1:])#now independent variables are column index 1-784
        sub[int(line[0]-1)]=1
        Ys.append(sub) #dependent variable at i=0
        
    num_features = len(Xs[0])
    num_labels = len(Ys[0])   
    train_size = len(Xs)    
    x_data = np.matrix(Xs)
    y_data = np.matrix(Ys)    
    X = tf.constant(x_data, dtype=DTYPE)    
    Y = tf.constant(y_data, dtype=DTYPE)    
    w = tf.Variable(tf.zeros([num_features, num_labels]))    
    b = tf.Variable(tf.zeros([num_labels]))

    return X, Y, w, b, num_features, num_labels, train_size

def multiclass_classifier(data, xs, labels, w, b, num_features, num_labels, train_size):
    learning_rate = 0.01
    training_epochs = 1000 #was 1000 use 1 to run quick
    momentum = 0.0
    batch_size = 100
    
    y_model = lambda: math.softmax(tf.matmul(X, w) + b) #use this instead of 
    cost = lambda: - math.reduce_sum(Y * math.log(y_model())) #generalization of the fun of binary classifier for many classes, multiply element by element not by matrices
    #
    train_op = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    
    for step in range(training_epochs * train_size // batch_size):
        offset = (step * batch_size) % train_size
        batch_xs = xs[offset:(offset + batch_size), :]
        batch_labels = labels[offset:(offset + batch_size)]
        X = tf.constant(batch_xs, dtype=DTYPE)
        Y = tf.constant(batch_labels, dtype=DTYPE)
        train_op.minimize(cost, [w, b])
        err = cost().numpy()
        print(step, err)
    #repeat for 3 batches> 3000 times
            
    w_val = w.numpy()
    b_val = b.numpy()
    print(w_val, b_val)
    
    w_val #length is number of features 700sth
    b_val
    
    predicted_ys = tf.argmax(y_model(), 1)
    print(predicted_ys)
    correct_prediction = lambda: tf.equal(tf.argmax(y_model(), 1), tf.argmax(Y, 1))
    #argmax finds position of the maximum
    #gives classification
    #tell it row wise 1
    # column wise 0
    #makes columns of position of maximum
    #[2] #position at index 2 has max
    #[1]
    #[1]
    #...
    #now you have two lists your Ys and the y_model
    #cmomparison returns list with boolean values
    accuracy = tf.reduce_mean(tf.cast(correct_prediction(), "float"))
    print('accuracy: ', accuracy.numpy())
    return w, b, predicted_ys
 
def calc_matrix(Y, predY):
    """We have 16 possible combinations"""
    t11, f12, f13, f14, f21,t22, f23, f24, f31, f32, t33, f34, f41, f42, f43, t44 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    i=0
    pred_Y = predY.numpy()
    print(pred_Y)
    Y = tf.argmax(Y, 1).numpy()
    
    for i in range(0, len(predY)):
        
        if (Y[i]==0 and pred_Y[i]==0):
            t11+=1
        elif (Y[i]==0 and pred_Y[i]==1):
            f12+=1
        elif (Y[i]==0 and pred_Y[i]==2):
            f13+=1  
        elif (Y[i]==0 and pred_Y[i]==3):
            f14+=1
            
        elif (Y[i]==1 and pred_Y[i]==0):
            f21+=1
        elif (Y[i]==1 and pred_Y[i]==1):
            t22+=1
        elif (Y[i]==1 and pred_Y[i]==2):
            f23+=1  
        elif (Y[i]==1 and pred_Y[i]==3):
            f24+=1   
            
        elif (Y[i]==2 and pred_Y[i]==0):
            f31+=1
        elif (Y[i]==2 and pred_Y[i]==1):
            f32+=1
        elif (Y[i]==2 and pred_Y[i]==2):
            t33+=1  
        elif (Y[i]==2 and pred_Y[i]==3):
            f34+=1 

        elif (Y[i]==3 and pred_Y[i]==0):
            f41+=1
        elif (Y[i]==3 and pred_Y[i]==1):
            f42+=1
        elif (Y[i]==3 and pred_Y[i]==2):
            f43+=1  
        elif (Y[i]==3 and pred_Y[i]==3):
            t44+=1         
    
    return t11, f12, f13, f14, f21,t22, f23, f24, f31, f32, t33, f34, f41, f42, f43, t44
  
def table(t11, f12, f13, f14, f21,t22, f23, f24, f31, f32, t33, f34, f41, f42, f43, t44):
    print ("{:<30}{:<30}".format("Confusion Matrix:", "Predicted"))
    print ("{:<30}{:<15}{:<15}{:<15}{:<15}".format("", "Label 1", "Label 2", "Label 3", "Label 4"))
    print()
    print (("{:<15}"*6).format("actual:", "Label 1", t11, f12, f13, f14))
    print()
    print (("{:<15}"*6).format("", "Label 2", f21,t22, f23, f24))
    print()
    print (("{:<15}"*6).format("", "Label 3", f31, f32, t33, f34))
    print()
    print (("{:<15}"*6).format("", "Label 4", f41,f42, f43, t44))
    print()
    #precision = tp/(tp+fp) #score of how likely a positive prediction is to be correct
    #recall = tp/(tp+fn) #t measures the ratio of true positives found. It’s is a score of how many true positives were successfully predicted    
    
    #print(">> The precision is: {}".format(round(precision, 4)))
    #print(">> The recall is: {}".format(round(recall, 4)))
    
def main():
    #get data from file
    training_data, testing_data = get_data()
    #initialize data
    X, Y, w, b, num_features, num_labels, train_size = initialize_data(training_data)
    #do multiclass classification
    w, b, predicted_ys = multiclass_classifier(training_data, X, Y, w, b, num_features, num_labels, train_size)
    #calculate matric values
    t11, f12, f13, f14, f21,t22, f23, f24, f31, f32, t33, f34, f41, f42, f43, t44 = calc_matrix(Y, predicted_ys)
    #display confusion matrix
    table(t11, f12, f13, f14, f21,t22, f23, f24, f31, f32, t33, f34, f41, f42, f43, t44)
main() 
