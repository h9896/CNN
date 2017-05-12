# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:46:13 2017

@author: Edison Song
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 01:38:19 2017

@author: Edison Song
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
#print(mnist.train.num_examples)


filenamep = "C:/Users/Edison Song/Desktop/testdata/da/nordata.csv"
filenametar = "C:/Users/Edison Song/Desktop/testdata/da/nortar.csv"
nor = np.float32(np.genfromtxt(filenamep, delimiter=','))
nort = np.float64(np.genfromtxt(filenametar, delimiter=','))
filenamep = "C:/Users/Edison Song/Desktop/testdata/da/abnordata.csv"
filenametar = "C:/Users/Edison Song/Desktop/testdata/da/abnortar.csv"
abnor = np.float32(np.genfromtxt(filenamep, delimiter=','))
abnort = np.float64(np.genfromtxt(filenametar, delimiter=','))
filenamep = "C:/Users/Edison Song/Desktop/testdata/da/test_all.csv"
filenametar = "C:/Users/Edison Song/Desktop/testdata/da/tar_all.csv"
test_all = np.float32(np.genfromtxt(filenamep, delimiter=','))
test_all_t = np.float64(np.genfromtxt(filenametar, delimiter=','))



n_classes = 5
batch_size = 100

x = tf.placeholder('float', [None, 44])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):
    weights = {
        # 5 x 5 convolution, 1 input image, 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs 
        'W_conv2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
        'W_conv3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'W_fc': tf.Variable(tf.random_normal([2*128, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_conv3': tf.Variable(tf.random_normal([128])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # Reshape input to a 4D tensor 
    x = tf.reshape(x, shape=[-1, 11, 4, 1])
    # Convolution Layer, using our function
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)
    conv1 = tf.nn.local_response_normalization(conv1)
#    conv1 = tf.nn.dropout(conv1, keep_prob)
    # Convolution Layer
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)
    conv2 = tf.nn.local_response_normalization(conv2)
#    conv2 = tf.nn.dropout(conv2, keep_prob)
    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3)
    conv3 = tf.nn.local_response_normalization(conv3)
#    conv3 = tf.nn.dropout(conv3, keep_prob)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer
    fc = tf.reshape(conv3, [-1, 2*128])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)
    
    output = tf.matmul(fc, weights['out']) + biases['out']
    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    avg_set=[]
    epoch_set=[]
    hm_epochs = 150
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(3):
                li_nor = random.sample(range(364),80)
                li_abnor = random.sample(range(49),20)
                paramter=[]
                for i in li_nor:
                    paramter.append(nor[i])
                for i in li_abnor:
                    paramter.append(abnor[i])
                paramter = np.array(paramter)
                results=[]
                for i in li_nor:
                    results.append(nort[i])
                for i in li_abnor:
                    results.append(abnort[i])
                results = np.array(results)
                epoch_x, epoch_y = paramter, results
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            acc = sess.run(accuracy, feed_dict={x: epoch_x,  y: epoch_y})
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss, ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            epoch_set.append(epoch)
            avg_set.append(epoch_loss)
        plt.plot(epoch_set,avg_set,'o', label='Logistic Regression Training phase')
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.show()
        print('Accuracy:',accuracy.eval({x:test_all, y:test_all_t}))
#        print("Testing Sensitivity:",accuracy.eval({x:paramter_tt, y:results_tt}))
        saver=tf.train.Saver()
        save_path = saver.save(sess, "C:/Users/Edison Song/Desktop/tensorflowtest/model12/model.ckpt")
        print("Model save in file: %s" % save_path)
        yy=sess.run(prediction,feed_dict={x:test_all})
        r=[]
        for i in yy:
            temp=[]
            temp.append(i[0])
            temp.append(i[1])
            temp.append(i[2])
            temp.append(i[3])
            temp.append(i[4])
            r.append(temp.index(max(temp)))
    plt.plot(r,'o')
    plt.show()
    return r
r=train_neural_network(x)