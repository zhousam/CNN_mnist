from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 导入CNN模块
import CNN_model
model = CNN_model.CNNmodel()
xs = model.xs
keep_prob = model.keep_prob

# 加载保存的训练模型
import global_variable
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess,global_variable.save_path)

# 测试精度
TestSize = 2
accuracyTotal = np.zeros((TestSize, 1), dtype=np.float32)
for j in range(TestSize):
    batch_xs, batch_ys = mnist.test.next_batch(int(10000/TestSize))
    accuracyTotal[j] = sess.run(model.compute_accuracy(batch_ys),feed_dict={xs:batch_xs,keep_prob:1})
print('>> Result T-accuracy = ', accuracyTotal.sum()/TestSize)
