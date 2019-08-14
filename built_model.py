# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)#E:/nuts/20180502_personal/learn/DeepLearning/

tf.reset_default_graph()
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data维度 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 数字=> 10 classes
# Set model weights
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 构建测试模型
s1 = tf.matmul(x, W) + b
pred = tf.nn.softmax(s1)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
model_path = "log/521model.ckpt"
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_path)
    Accaa = sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print ("Accuracy:", Accaa)