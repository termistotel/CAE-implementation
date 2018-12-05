import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

def displayRec(images):
	asdf = sess.run(out, feed_dict={x :images})
	for i in asdf:
		plt.imshow(i)
		plt.show()

imgs = np.array([cv2.cvtColor(cv2.resize(cv2.imread("data/"+file), (0,0), fx=0.02, fy=0.02), cv2.COLOR_BGR2RGB) for file in os.listdir("data")])/255
N = np.prod(imgs.shape[1:])

base = 32
filternum = range(4)
filternum = list(map(lambda x: base * 2**x, filternum))
filternum = [32, 64, 64, 128]
graph1 = tf.Graph()
with graph1.as_default():
	x = tf.placeholder(shape= ([None]+list(imgs.shape[1:])), dtype=tf.float32)
	alfa = tf.placeholder(shape=(), dtype=tf.float32)
	lambdad = tf.placeholder(shape=(), dtype=tf.float32)

	gs = tf.Variable(0, trainable=False)

	with tf.variable_scope("conv1"):
		l = tf.layers.conv2d(x, filternum[0], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		l = tf.layers.conv2d(l, filternum[0], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		l = tf.layers.dropout(l)
		m1 = tf.layers.max_pooling2d(l, (2,2), (2,2), padding="SAME")

	with tf.variable_scope("conv2"):
		l = tf.layers.conv2d(m1, filternum[1], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		l = tf.layers.conv2d(l, filternum[1], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		l = tf.layers.dropout(l)
		m2 = tf.layers.max_pooling2d(l, (2,2), (2,2), padding="SAME")

	with tf.variable_scope("conv3"):
		l = tf.layers.conv2d(m2, filternum[2], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		l = tf.layers.conv2d(l, filternum[2], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		l = tf.layers.dropout(l)
		m3 = tf.layers.max_pooling2d(l, (2,2), (2,2), padding="SAME")

	with tf.variable_scope("conv4"):
		l = tf.layers.conv2d(m3, filternum[3], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		l = tf.layers.conv2d(l, filternum[3], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		l = tf.layers.dropout(l)
		# m4 = tf.layers.max_pooling2d(l, (2,2), (2,2), padding="SAME")

	# with tf.variable_scope("dense"):
	# 	flat= tf.layers.flatten(m4)
	# 	N=np.prod(m4.get_shape().as_list()[1:])
	# 	dense = tf.layers.dense(flat, N//10, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
	# 	dense = tf.layers.dense(dense, N, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
	# 	unflat = tf.reshape(dense, tf.shape(m4))

	# with tf.variable_scope("upscale1"):
	# 	u = tf.image.resize_images(unflat, size=tf.shape(m3)[1:3])
	# 	l = tf.layers.conv2d(u, filternum[2], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
	# 	l = tf.layers.conv2d(l, filternum[2], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))

	with tf.variable_scope("upscale2"):
		u = tf.image.resize_images(l, size=tf.shape(m2)[1:3])
		l = tf.layers.conv2d(u, filternum[1], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		l = tf.layers.conv2d(l, filternum[1], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		l = tf.layers.dropout(l)

	with tf.variable_scope("upscale3"):
		u = tf.image.resize_images(l, size=tf.shape(m1)[1:3])
		l = tf.layers.conv2d(u, filternum[0], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		l = tf.layers.conv2d(l, filternum[0], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		l = tf.layers.dropout(l)

	with tf.variable_scope("upscale4"):
		u = tf.image.resize_images(l, size=tf.shape(x)[1:3])
		out = tf.layers.conv2d(u, 3, (3,3), (1,1), padding="SAME", activation=tf.nn.sigmoid)


	learning_rate = tf.train.exponential_decay(alfa,gs,10000,0.1)
	loss = tf.losses.mean_squared_error(x, out) + tf.losses.get_regularization_loss()
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	optimize = optimizer.minimize(loss, global_step = gs)
	grads = optimizer.compute_gradients(loss)

	gradSummaries = []
	for g,v in grads:# tf.trainable_variables():
		if not g is None:
			with tf.name_scope("grads"):
				gnorm = tf.sqrt(tf.reduce_mean(tf.square(g)))
				gradSummaries.append(tf.summary.scalar(v.name, gnorm))
		else:
			print("eto: ", v.name, g)
			input()
	gSummaries = tf.summary.merge(gradSummaries)


	with tf.name_scope("loss_data"):
		testLoss = tf.placeholder(tf.float32, shape=None, name="test_loss_summary")
		trainLoss = tf.placeholder(tf.float32, shape=None, name="train_loss_summary")
		testLossSummary = tf.summary.scalar("TestLoss", testLoss)
		trainLossSummary = tf.summary.scalar("TrainLoss", trainLoss)
		logTestLossSummary = tf.summary.scalar("TestLossLog", tf.log(testLoss))
		logTrainLossSummary = tf.summary.scalar("TrainLossLog", tf.log(trainLoss))
	lossSummaries = tf.summary.merge([testLossSummary, logTestLossSummary, trainLossSummary, logTrainLossSummary])


config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(graph=graph1, config=config)


summ_writer = tf.summary.FileWriter("summaries", sess.graph)

# # with tf.Session(graph=graph1) as sess:
sess.run(tf.global_variables_initializer())

np.random.shuffle(imgs)
img = imgs[:-5]
test = imgs[-5:]

alpha = 0.0005
lam = 0.0
batchsize = 4
M = img.shape[0]

for i in range(1000):
	print("epoch: ", i)
	np.random.shuffle(img)
	for j in range(M//batchsize):
		sess.run(optimize, feed_dict={x: img[j*batchsize:(j+1)*batchsize], alfa: alpha, lambdad: lam})
	# print("LR: ", sess.run(learning_rate, feed_dict={alfa:alpha}))
	print("train: ", sess.run(loss, feed_dict={x: img, lambdad: lam}))
	print("test: ", sess.run(loss, feed_dict={x: test, lambdad: lam}))
	summ1 = sess.run(lossSummaries, feed_dict={ 
							trainLoss: sess.run(loss, feed_dict={x:img, lambdad: lam}),
							testLoss: sess.run(loss, feed_dict={x:test, lambdad: lam}) 
							})
	summ2 = sess.run(gSummaries, feed_dict={x: img[j*batchsize:(j+1)*batchsize], alfa:alpha, lambdad: lam})
	summ_writer.add_summary(summ2, i)
	summ_writer.add_summary(summ1, i)

for i, val in enumerate(sess.run(out, feed_dict={x: img[-5:]})):
	plt.imshow((img[-5:])[i])
	plt.show()
	plt.imshow(val)
	plt.show()

for i, val in enumerate(sess.run(out, feed_dict={x: test})):
	plt.imshow(test[i])
	plt.show()
	plt.imshow(val)
	plt.show()
