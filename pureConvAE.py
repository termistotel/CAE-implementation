import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

imgs = np.array([cv2.cvtColor(cv2.resize(cv2.imread("data/"+file), (0,0), fx=0.1, fy=0.1), cv2.COLOR_BGR2RGB) for file in os.listdir("data")])

graph1 = tf.Graph()
with graph1.as_default():
	x = tf.placeholder(shape= ([None]+list(imgs.shape[1:])), dtype=tf.float32)
	X=x/255
	alfa = tf.placeholder(shape=(), dtype=tf.float32)
	# ConvLayer 1
	l1 = tf.layers.conv2d(X, 64, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
	l1 = tf.layers.conv2d(l1, 64, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
	# l1 = tf.layers.dropout(l1)
	# Maxpool
	m1 = tf.nn.max_pool(l1, (1,2,2,1), (1,2,2,1), "SAME")
	# ConvLayer 2
	l2 = tf.layers.conv2d(m1, 128, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
	l2 = tf.layers.conv2d(l2, 128, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
	# l2 = tf.layers.dropout(l2)
	# Maxpool
	m2 = tf.nn.max_pool(l2, (1,2,2,1), (1,2,2,1), "SAME")
	# ConvLayer 3
	l3 = tf.layers.conv2d(m2, 256, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
	l3 = tf.layers.conv2d(l3, 256, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
	# l3 = tf.layers.dropout(l3)
	# Maxpool
	m3 = tf.nn.max_pool(l3, (1,2,2,1), (1,2,2,1), "SAME")
	# ConvLayer 4
	l4 = tf.layers.conv2d(m3, 512, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
	l4 = tf.layers.conv2d(l4, 512, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
	# Upsample 4
	u4 = tf.image.resize_images(l4, size=tf.shape(m2)[1:3])
	l4 = tf.layers.conv2d(u4, 128, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
	# l4 = tf.layers.dropout(l4)
	# Upsample 5
	u5 = tf.image.resize_images(l4, size=tf.shape(m1)[1:3])
	l5 = tf.layers.conv2d(u5, 64, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
	# l5 = tf.layers.dropout(l5)
	# Upsample 6
	u6 = tf.image.resize_images(l5, size=tf.shape(X)[1:3])
	out = tf.layers.conv2d(u6, 3, (3,3), (1,1), padding="SAME", activation=tf.nn.sigmoid)
	loss = tf.reduce_mean(tf.square(out - X))
	optimize = tf.train.AdamOptimizer(learning_rate=alfa).minimize(loss)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(graph=graph1, config=config)
# # with tf.Session(graph=graph1) as sess:
sess.run(tf.global_variables_initializer())

np.random.shuffle(imgs)
img = imgs[:-5]
test = imgs[-5:]
M = img.shape[0]
batchsize = 4
for i in range(300):
	print("epoch: ", i)
	np.random.shuffle(img)
	for j in range(int(M/batchsize)):
		sess.run(optimize, feed_dict={x: img[j*batchsize:(j+1)*batchsize], alfa:0.00001})
	print("train: ", sess.run(loss, feed_dict={x: img}))
	print("test: ", sess.run(loss, feed_dict={x: test}))

for i in sess.run(out, feed_dict={x: test}):
	plt.imshow(i)
	plt.show()
