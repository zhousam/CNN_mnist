from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import global_variable

import CNN_model

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

model = CNN_model.CNNmodel()
xs = model.xs
ys = model.ys
keep_prob = model.keep_prob

# 创建会话，初始化
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 开始训练
iTrain = 1
AccuracyGet = []
accuracyTotalV = np.zeros((1, 1), np.float32)
for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    sess.run(model.train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 54 == 0:
        accuracyTotalV = sess.run(model.compute_accuracy(mnist.validation.labels),
                                  feed_dict={xs: mnist.validation.images, keep_prob: 1})
        AccuracyGet.append(accuracyTotalV)
        print('>> Time = ', iTrain, ' V-accuracy = ', accuracyTotalV)
        iTrain += 1
        if accuracyTotalV > 0.99:
            break

# 保存模型
saver = tf.train.Saver()
saver.save(sess, global_variable.save_path)
sess.close()
print("模型已经保存->->->")

# 打印训练精度
import matplotlib.pyplot as plt
plt.plot(np.arange(len(AccuracyGet)), AccuracyGet)
plt.xlabel("Train Time")
plt.ylabel("Train Accuracy")
plt.show()