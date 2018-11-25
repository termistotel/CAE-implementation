import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2

img = np.array([cv2.cvtColor(cv2.resize(cv2.imread("data/"+file), (0,0), fx=0.2, fy=0.2), cv2.COLOR_BGR2RGB) for file in os.listdir("data")])

graph1 = tf.Graph()
with graph1.as_default():
	x = tf.placeholder(shape= ([None]+list(img.shape[1:])), dtype=tf.float32)
	X=x/255
	alfa = tf.placeholder(shape=(), dtype=tf.float32)
	# ConvLayer 1
	w1 = tf.Variable( tf.random.truncated_normal(shape=(f1, f1, np.shape(img)[-1], 6), dtype=tf.float32) )
	b1 = tf.Variable( tf.random.truncated_normal(shape=(1,1,1,6), dtype=tf.float32))
	A1 = tf.nn.relu(tf.nn.conv2d( X, w1, strides=[1,1,1,1], padding="VALID" ) + b1)
	# Maxpool
	A15 = tf.layers.dropout(tf.nn.max_pool(A1, (1,2,2,1), (1,2,2,1), "VALID"))
	# Upsample
	A35 = tf.image.resize_images(A15, size=tf.shape(X)[1:3])
	f4 = 3
	w4 = tf.Variable( tf.random.truncated_normal(shape=(f4, f4, 6, np.shape(img)[-1]), dtype=tf.float32) )
	b4 = tf.Variable( tf.random.truncated_normal(shape=(1,1,1,np.shape(img)[-1]), dtype=tf.float32))
	A4 = tf.nn.sigmoid(tf.nn.conv2d( A35, w4, strides=[1,1,1,1], padding="SAME" ) + b4)
	loss = tf.reduce_mean(tf.square(A4 - X))
	optimize = tf.train.AdamOptimizer(learning_rate=alfa).minimize(loss)

sess = tf.InteractiveSession(graph=graph1)
# # with tf.Session(graph=graph1) as sess:
sess.run(tf.global_variables_initializer())
for i in range(10000):
	sess.run(optimize, feed_dict={x: img, alfa:0.01})
	print(sess.run(loss, feed_dict={x: img}))
