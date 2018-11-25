import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

imgs = np.array([cv2.cvtColor(cv2.resize(cv2.imread("data/"+file), (0,0), fx=0.05, fy=0.05), cv2.COLOR_BGR2RGB) for file in os.listdir("data")])/255
N = np.prod(imgs.shape[1:])

base = 6
filternum = range(4)
filternum = list(map(lambda x: base * 2**x, filternum))

lambdac, lambdad = 0.001, 0.001

graph1 = tf.Graph()
with graph1.as_default():
	x = tf.placeholder(shape= ([None]+list(imgs.shape[1:])), dtype=tf.float32)
	# X=x/255
	X = x
	alfa = tf.placeholder(shape=(), dtype=tf.float32)
	gs = tf.Variable(0, trainable=False)

	# ConvLayer 1
	with tf.variable_scope("conv1"):
		l1 = tf.layers.conv2d(X, filternum[0], (7,7), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		# l1 = tf.layers.conv2d(l1, filternum[0], (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
		# l1 = tf.layers.dropout(l1)
		# Maxpool
		m1 = tf.nn.max_pool(l1, (1,2,2,1), (1,2,2,1), "SAME")

	# ConvLayer 2
	with tf.variable_scope("conv2"):
		l2 = tf.layers.conv2d(m1, filternum[1], (5,5), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		# l2 = tf.layers.conv2d(l2, 128, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
		# l2 = tf.layers.dropout(l2)
		# Maxpool
		m2 = tf.nn.max_pool(l2, (1,2,2,1), (1,2,2,1), "SAME")

	# ConvLayer 3
	with tf.variable_scope("conv3"):
		l3 = tf.layers.conv2d(m2, filternum[2], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		# l3 = tf.layers.conv2d(l3, 256, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
		# l3 = tf.layers.dropout(l3)
		# Maxpool
		m3 = tf.nn.max_pool(l3, (1,2,2,1), (1,2,2,1), "SAME")
	# ConvLayer 4
	with tf.variable_scope("conv4"):
		l4 = tf.layers.conv2d(m3, filternum[3], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		m4 = tf.nn.max_pool(l4, (1,2,2,1), (1,2,2,1), "SAME")

	with tf.variable_scope("flatten"):
		flat1 = tf.layers.flatten(m4)
		N=np.prod(m4.get_shape().as_list()[1:])

	# DenseShit
	with tf.variable_scope("dense1"):
		dense1 = tf.layers.dense(flat1, int(N/10), activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		# dense1 = tf.layers.dropout(dense1)

	with tf.variable_scope("dense2"):
		dense2 = tf.layers.dense(dense1, int(N/20), activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		# dense2 = tf.layers.dropout(dense2)

	with tf.variable_scope("dense3"):
		dense3 = tf.layers.dense(dense2, int(N/40), activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		# dense3 = tf.layers.dropout(dense3)

	with tf.variable_scope("dense4"):
		dense4 = tf.layers.dense(dense3, int(N/20), activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		# dense4 = tf.layers.dropout(dense4)

	with tf.variable_scope("dense5"):
		dense5 = tf.layers.dense(dense4, int(N/10), activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		# dense5 = tf.layers.dropout(dense5)

	with tf.variable_scope("Unflat"):
		unflat = tf.reshape(tf.layers.dense(dense5, N, activation=tf.nn.relu), tf.shape(m3))

	# with tf.variable_scope("conv5"):
	# 	l4 = tf.layers.conv2d(unflat, filternum[3], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))

	with tf.variable_scope("Upscale1"):
		# Upsample 1
		u5 = tf.image.resize_images(l4, size=tf.shape(m3)[1:3])
		l5 = tf.layers.conv2d(u5, filternum[2], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		# l5 = tf.layers.dropout(l5)

	with tf.variable_scope("Upscale2"):
		# Upsample 2
		u6 = tf.image.resize_images(l5, size=tf.shape(m2)[1:3])
		l6 = tf.layers.conv2d(u6, filternum[1], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		# l6 = tf.layers.dropout(l6)

	with tf.variable_scope("Upscale3"):
		# Upsample 4
		u7 = tf.image.resize_images(l6, size=tf.shape(m1)[1:3])
		l7 = tf.layers.conv2d(u7, filternum[0], (5,5), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		# l7 = tf.layers.dropout(l7)

	with tf.variable_scope("Upscale4"):
		# Upsample 4
		u8 = tf.image.resize_images(l7, size=tf.shape(X)[1:3])
		l8 = tf.layers.conv2d(u8, 3, (7,7), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		# l8 = tf.layers.dropout(l8)

	with tf.variable_scope("outlayer"):
		out = tf.layers.conv2d(l8, 3, (7,7), (1,1), padding="SAME", activation=tf.nn.sigmoid, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))

	loss = tf.losses.mean_squared_error(X, out) + tf.losses.get_regularization_loss()
	optimizer = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(alfa,gs,100000,0.1))
	grads = optimizer.compute_gradients(loss)
	optimize = optimizer.minimize(loss, global_step = gs)

	gradSummaries = []
	for g,v in grads:
		with tf.name_scope("grads"):
			gnorm = tf.sqrt(tf.reduce_mean(tf.square(g)))
			gradSummaries.append(tf.summary.scalar(v.name, gnorm))
	gSummaries = tf.summary.merge(gradSummaries)

	with tf.name_scope("loss_data"):
		testLoss = tf.placeholder(tf.float32, shape=None, name="test_loss_summary")
		trainLoss = tf.placeholder(tf.float32, shape=None, name="train_loss_summary")

		testLossSummary = tf.summary.scalar("Test loss", testLoss)
		trainLossSummary = tf.summary.scalar("Train loss", trainLoss)

		logTestLossSummary = tf.summary.scalar("Test loss log", testLoss)
		logTrainLossSummary = tf.summary.scalar("Train loss log", trainLoss)

	lossSummaries = tf.summary.merge([testLossSummary, logTestLossSummary, trainLossSummary, logTrainLossSummary])


config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(graph=graph1, config=config)

def displayRec(images):
	asdf = sess.run(out, feed_dict={x :images})
	for i in asdf:
		plt.imshow(i)
		plt.show()

summ_writer = tf.summary.FileWriter("summaries", sess.graph)

# # with tf.Session(graph=graph1) as sess:
sess.run(tf.global_variables_initializer())

np.random.shuffle(imgs)
img = imgs[:-5]
test = imgs[-5:]
M = img.shape[0]
batchsize = 1
for i in range(10000):
	print("epoch: ", i)
	np.random.shuffle(img)

	for j in range(int(M/batchsize)):
		sess.run(optimize, feed_dict={x: img[j*batchsize:(j+1)*batchsize], alfa:0.0005})

	# print("train: ", sess.run(loss, feed_dict={x: img}))
	# print("test: ", sess.run(loss, feed_dict={x: test}))

	summ1 = sess.run(lossSummaries, feed_dict={ 
							trainLoss: sess.run(loss, feed_dict={x:img}),
							testLoss: sess.run(loss, feed_dict={x:test}) 
							})

	summ2 = sess.run(gSummaries, feed_dict={x: img[j*batchsize:(j+1)*batchsize], alfa:0.0005})

	summ_writer.add_summary(summ1, i)
	summ_writer.add_summary(summ2, i)

for i in sess.run(out, feed_dict={x: img[-5:]}):
	plt.imshow(i)
	plt.show()

for i in sess.run(out, feed_dict={x: test}):
	plt.imshow(i)
	plt.show()
