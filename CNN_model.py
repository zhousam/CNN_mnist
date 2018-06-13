from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.set_random_seed(1)

class CNNmodel:
    # 定义权重
    @staticmethod
    def __weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    # 定义偏置
    @staticmethod
    def __bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 定义步长strides=[1,1,1,1]值，strides[0]和strides[3]的两个1是默认值
    # --> stride [1, x_movement, y_movement, 1]
    # --> Must have strides[0] = strides[3] = 1
    # --> 中间两个1代表移动时在x方向运动一步，y方向运动一步
    # padding采用的方式是SAME。
    @staticmethod
    def __conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # 池化的核函数大小为2x2，因此 ksize =[1,2,2,1]，步长 stride=2
    @staticmethod
    def __max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    def compute_accuracy(self, v_ys):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def __init__(self):
        # define placeholder for inputs to network
        self.xs = tf.placeholder(tf.float32, [None, 784])/255.   # 784=28x28
        self.ys = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.x_image = tf.reshape(self.xs, [-1, 28, 28, 1])
        # print(x_image.shape)  # will show --> [n_samples_BatchSize, 28,28,1]

        ## layer 1: conv_1  ##
        self.W_conv1 = self.__weight_variable([5,5, 1,32])  # filter_size:5x5, in_size:1(gray image),  out_size:32(filter number)
        self.b_conv1 = self.__bias_variable([32])
        self.h_conv1 = tf.nn.relu(self.__conv2d(self.x_image, self.W_conv1) + self.b_conv1) # output size 28x28x32
        self.h_pool1 = self.__max_pool_2x2(self.h_conv1)    # output size 14x14x32

        ## layer 2: conv_2  ##
        self.W_conv2 = self.__weight_variable([5,5, 32, 64]) # filter_size:5x5, in_size:32,  out_size:64
        self.b_conv2 = self.__bias_variable([64])
        self.h_conv2 = tf.nn.relu(self.__conv2d(self.h_pool1, self.W_conv2) + self.b_conv2) # output size 14x14x64
        self.h_pool2 = self.__max_pool_2x2(self.h_conv2)     # output size 7x7x64

        ## layer 3: fc_1  ##
        self.W_fc1 = self.__weight_variable([7*7*64, 1024])
        self.b_fc1 = self.__bias_variable([1024])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])  # [n_samples_BatchSize,7,7,64] --> [n_samples_BatchSize,7*7*64]
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1) # size:[n_samples_BatchSize,1024]
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        ## layer 4: fc_2  ##
        self.W_fc2 = self.__weight_variable([1024, 10])
        self.b_fc2 = self.__bias_variable([10])
        self.prediction = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)  # size:[n_samples_BatchSize,10]

        # the error between prediction and real data
        # reduction_indices：确定维度
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction), reduction_indices=[1])) # 交叉熵损失
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
