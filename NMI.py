# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 13:50:23 2018

@author: hp
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def read_dataset():
    df = pd.read_csv("E:\Python Exercises\Deeplearning_tutorial\sonar.csv")
    X = df[df.columns[0:60]].values
    y = df[df.columns[60]]
    
    #Encode dependent variables
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    print(X.shape)
    return(X, Y)
    
#Defining one_hot_encode function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

#Reading the dataset
X, Y = read_dataset()

#Shuffle dataset to mix the rows
X, Y = shuffle(X, Y, random_state=1)

#Train and test split
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state = 415)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)

#Important parameters of the tensors
learning_rate = 0.3
training_epochs = 1000
cost_history = np.empty(shape = [1], dtype = float)
n_dim = X.shape[1]
print("n_dim", n_dim)
n_class = 2
model_path = "E:\Python Exercises\Deeplearning_tutorial"

#Defining number of hidden layer and neurons for each layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32,[None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])

#Defining the model
def multilayer_perceptron(x, weights, biases):
    #Hidden layer with sigmoid
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    #Hidden layer with sigmoid
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    #Hidden layer with sigmoid
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    #Hidden layer with Relu
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

#Output lyer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

#Defining weights and biases for each layer
weights = {
        'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
        'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
        }

biases = {
        'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
        'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
        'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
        'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
        'out': tf.Variable(tf.truncated_normal([n_class]))
        }

#Initialize all variables

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#Call the model defined
y = multilayer_perceptron(x, weights, biases)

#Define cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
#logits=y is model output and labels=y_ is actual output
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)


sess = tf.Session()
sess.run(init) 

#Calculate cost and accuracy for each epoch
mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={x:train_x, y_:train_y})
    cost = sess.run(cost_function, feed_dict={x:train_x, y_:train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))#to see diff between actual and predicted output 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    pred_y = sess.run(y, feed_dict={x: test_x})#feeding test data and finding how accurate our model is
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x:train_x, y_:train_y}))#accuracy on train data not on test data
    accuracy_history.append(accuracy)
    
    print("epoch:", epoch, "-", "cost:", cost, "-MSE:", mse_, "-Train accuracy:", accuracy)
    
save_path = saver.save(sess, model_path)
print("Model saved in file:%s" % save_path)

#Plot MSE and accuracy graph
plt.plot(mse_history, 'r')
plt.show()
plt.plot(accuracy_history)
plt.show()

#Print final accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test accuracy:", (sess.run(accuracy, feed_dict={x:test_x, y_:test_y})))

#Print final mean square error
pred_y = sess.run(y, feed_dict={x:test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE:%.4f" %sess.run(mse))

#To check how good the prediction is
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init) 
saver.restore(sess, model_path)

prediction = tf.argmax(y,1)
correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('********************************************************************')
print('0 -> Mine(M) and 1->Rock(R)')
print('*********************************************************************')
for i in range(93, 101):
    prediction_run = sess.run(prediction, feed_dict={x: X[i].reshape(1,60)})
    accuracy_run = sess.run(accuracy, feed_dict={x:X[i].reshape(1,60), y_:test_y[i]})
    print("Original class:", y[i], "Predicted values:", prediction_run[i])