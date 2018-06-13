# Learning From -- 1 code: https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/401_CNN.py
# Learning From -- 2 mnist Message://blog.csdn.net/zchang81/article/details/61918577
# Learning From -- 3 CNN Base Knowledge Video: http://mooc.study.163.com/course/2001281004?tid=2001392030#/info
# """
# Attention -> This code is only for python 3+.
# """

from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 获取数据维度
print ( "训练集的张量:", mnist.train.images.shape )
print ( "训练集标签的张量:", mnist.train.labels.shape )
print ( "验证集的张量:", mnist.validation.images.shape )
print ( "验证集标签的张量:", mnist.validation.labels.shape )
print ( "测试集的张量:", mnist.test.images.shape )
print ( "测试集标签的张量:", mnist.test.labels.shape )

# 定义权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
# 定义偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义步长strides=[1,1,1,1]值，strides[0]和strides[3]的两个1是默认值
# --> stride [1, x_movement, y_movement, 1]
# --> Must have strides[0] = strides[3] = 1
# --> 中间两个1代表移动时在x方向运动一步，y方向运动一步
# padding采用的方式是SAME。
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化的核函数大小为2x2，因此 ksize =[1,2,2,1]，步长 stride=2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255.   # 784=28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # will show --> [n_samples_BatchSize, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32])  # filter_size:5x5, in_size:1(gray image),  out_size:32(filter number)
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)    # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # filter_size:5x5, in_size:32,  out_size:64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)     # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  # [n_samples_BatchSize,7,7,64] --> [n_samples_BatchSize,7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # size:[?,1024]
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # size:[?,10]

# the error between prediction and real data
#  reduction_indices：确定维度
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])) # 交叉熵损失
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 计算精度
AccuracyGet = []
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs,  ys: v_ys, keep_prob: 1})
    return result

# 创建会话，初始化
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 开始训练
iTrain = 1
accuracyTotalV = np.zeros((1, 1), np.float32)
for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 54 == 0:
        accuracyTotalV = compute_accuracy(mnist.validation.images, mnist.validation.labels)
        AccuracyGet.append(accuracyTotalV)
        print('Time=', iTrain, ' V-accuracy = ', accuracyTotalV)
        iTrain += 1
        if accuracyTotalV > 0.99:
            break

# 测试精度
TestSize = 2
accuracyTotal = np.zeros((TestSize, 1), dtype=np.float32)
for j in range(TestSize):
    batch_xs, batch_ys = mnist.test.next_batch(int(10000/TestSize))
    accuracyTotal[j] = compute_accuracy(batch_xs, batch_ys)
print('T-accuracy = ', accuracyTotal.sum()/TestSize)

# 打印训练精度
import matplotlib.pyplot as plt
plt.plot(np.arange(len(AccuracyGet)), AccuracyGet)
plt.xlabel("Train Time")
plt.ylabel("Train Accuracy")
plt.show()