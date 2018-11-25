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

# lambdac, lambdad = 0.0001, 0.0001
lamb = 0.0001

graph1 = tf.Graph()
with graph1.as_default():
	with tf.device('/cpu:0'):
		x = tf.placeholder(shape= ([None]+list(imgs.shape[1:])), dtype=tf.float32)
		alfa = tf.placeholder(shape=(), dtype=tf.float32)
		gs = tf.Variable(0, trainable=False)

	# ConvLayer 1
	with tf.variable_scope("conv1"):
		# l1 = tf.layers.conv2d(X, filternum[0], (7,7), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		# l1 = tf.layers.conv2d(l1, filternum[0], (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
		# l1 = tf.layers.dropout(l1)
		w = tf.Variable( tf.random.truncated_normal(shape=(7, 7, 3, filternum[0]), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,1,1,filternum[0]), dtype=tf.float32))
		A = tf.nn.relu(tf.nn.conv2d( x, w, strides=[1,1,1,1], padding="SAME" ) + b)
		# A = tf.layers.dropout(A)
		# Maxpool
		m1 = tf.nn.max_pool(A, (1,2,2,1), (1,2,2,1), "SAME")

	# ConvLayer 2
	with tf.variable_scope("conv2"):
		# l2 = tf.layers.conv2d(m1, filternum[1], (5,5), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		# l2 = tf.layers.conv2d(l2, 128, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
		# l2 = tf.layers.dropout(l2)
		w = tf.Variable( tf.random.truncated_normal(shape=(5, 5, filternum[0], filternum[1]), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,1,1,filternum[1]), dtype=tf.float32))
		A = tf.nn.relu(tf.nn.conv2d( m1, w, strides=[1,1,1,1], padding="SAME" ) + b)
		# A = tf.layers.dropout(A)
		# Maxpool
		m2 = tf.nn.max_pool(A, (1,2,2,1), (1,2,2,1), "SAME")

	# ConvLayer 3
	with tf.variable_scope("conv3"):
		# l3 = tf.layers.conv2d(m2, filternum[2], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		# l3 = tf.layers.conv2d(l3, 256, (3,3), (1,1), padding="SAME", activation=tf.nn.relu)
		# l3 = tf.layers.dropout(l3)
		w = tf.Variable( tf.random.truncated_normal(shape=(3, 3, filternum[1], filternum[2]), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,1,1,filternum[2]), dtype=tf.float32))
		A = tf.nn.relu(tf.nn.conv2d( m2, w, strides=[1,1,1,1], padding="SAME" ) + b)
		# A = tf.layers.dropout(A)
		# Maxpool
		m3 = tf.nn.max_pool(A, (1,2,2,1), (1,2,2,1), "SAME")

	# ConvLayer 4
	with tf.variable_scope("conv4"):
		# l4 = tf.layers.conv2d(m3, filternum[3], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		w = tf.Variable( tf.random.truncated_normal(shape=(3, 3, filternum[2], filternum[3]), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,1,1,filternum[3]), dtype=tf.float32))
		A = tf.nn.relu(tf.nn.conv2d( m3, w, strides=[1,1,1,1], padding="SAME" ) + b)
		# A = tf.layers.dropout(A)
		# Maxpool
		m4 = tf.nn.max_pool(A, (1,2,2,1), (1,2,2,1), "SAME")
		
	with tf.variable_scope("flatten"):
		flat1 = tf.layers.flatten(m4)
		N=np.prod(m4.get_shape().as_list()[1:])

	# DenseShit
	with tf.variable_scope("dense1"):
		# dense1 = tf.layers.dense(flat1, int(N/10), activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		w = tf.Variable( tf.random.truncated_normal(shape = (N, N//10), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,N//10)))
		dense = tf.nn.relu(tf.matmul(flat1, w) + b)
		# dense1 = tf.layers.dropout(dense1)

	with tf.variable_scope("dense2"):
		# dense2 = tf.layers.dense(dense1, int(N/20), activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		w = tf.Variable( tf.random.truncated_normal(shape = (N//10, N//20), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,N//20)))
		dense = tf.nn.relu(tf.matmul(dense, w) + b)
		# dense2 = tf.layers.dropout(dense2)

	with tf.variable_scope("dense3"):
		# dense3 = tf.layers.dense(dense2, int(N/80), activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		w = tf.Variable( tf.random.truncated_normal(shape = (N//20, N//40), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,N//40)))
		dense = tf.nn.relu(tf.matmul(dense, w) + b)
		# dense3 = tf.layers.dropout(dense3)

	with tf.variable_scope("dense4"):
		# dense4 = tf.layers.dense(dense3, int(N/20), activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		w = tf.Variable( tf.random.truncated_normal(shape = (N//40, N//20), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,N//20)))
		dense = tf.nn.relu(tf.matmul(dense, w) + b)
		# dense4 = tf.layers.dropout(dense4)

	with tf.variable_scope("dense5"):
		# dense5 = tf.layers.dense(dense4, int(N/10), activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		w = tf.Variable( tf.random.truncated_normal(shape = (N//20, N//10), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,N//10)))
		dense = tf.nn.relu(tf.matmul(dense, w) + b)
		# dense5 = tf.layers.dropout(dense5)

	with tf.variable_scope("dense6"):
		# dense6 = tf.layers.dense(dense5, N, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdad))
		w = tf.Variable( tf.random.truncated_normal(shape = (N//10, N), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,N)))
		dense = tf.nn.relu(tf.matmul(dense, w) + b)
		# dense6 = tf.layers.dropout(dense6)

	with tf.variable_scope("Unflat"):
		unflat = tf.reshape(dense, tf.shape(m4))

	with tf.variable_scope("Upscale1"):
		# Upsample 1
		u = tf.image.resize_images(unflat, size=tf.shape(m3)[1:3])
		# l5 = tf.layers.conv2d(u5, filternum[2], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		# l5 = tf.layers.dropout(l5)
		w = tf.Variable( tf.random.truncated_normal(shape=(3, 3, filternum[3], filternum[2]), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,1,1,filternum[2]), dtype=tf.float32))
		A = tf.nn.relu(tf.nn.conv2d( u, w, strides=[1,1,1,1], padding="SAME" ) + b)

	with tf.variable_scope("Upscale2"):
		# Upsample 2
		u = tf.image.resize_images(A, size=tf.shape(m2)[1:3])
		# l6 = tf.layers.conv2d(u6, filternum[1], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		# l6 = tf.layers.dropout(l6)
		w = tf.Variable( tf.random.truncated_normal(shape=(3, 3, filternum[2], filternum[1]), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,1,1,filternum[1]), dtype=tf.float32))
		A = tf.nn.relu(tf.nn.conv2d( u, w, strides=[1,1,1,1], padding="SAME" ) + b)

	with tf.variable_scope("Upscale3"):
		# Upsample 3
		u = tf.image.resize_images(A, size=tf.shape(m1)[1:3])
		# l7 = tf.layers.conv2d(u7, filternum[0], (3,3), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		# l7 = tf.layers.dropout(l7)
		w = tf.Variable( tf.random.truncated_normal(shape=(5, 5, filternum[1], filternum[0]), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,1,1,filternum[0]), dtype=tf.float32))
		A = tf.nn.relu(tf.nn.conv2d( u, w, strides=[1,1,1,1], padding="SAME" ) + b)

	with tf.variable_scope("Upscale4"):
		# Upsample 3
		u = tf.image.resize_images(A, size=tf.shape(x)[1:3])
		w = tf.Variable( tf.random.truncated_normal(shape=(7, 7, filternum[0], 3), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,1,1, 3), dtype=tf.float32))
		A = tf.nn.relu(tf.nn.conv2d( u, w, strides=[1,1,1,1], padding="SAME" ) + b)


	with tf.variable_scope("outlayer"):
		# out = tf.layers.conv2d(l7, 3, (5,5), (1,1), padding="SAME", activation=tf.nn.sigmoid, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambdac))
		w = tf.Variable( tf.random.truncated_normal(shape=(7, 7, 3, 3), dtype=tf.float32) )
		b = tf.Variable( tf.zeros(shape=(1,1,1, 3), dtype=tf.float32))
		out = tf.nn.sigmoid(tf.nn.conv2d( A, w, strides=[1,1,1,1], padding="SAME" ) + b)

	loss = tf.losses.mean_squared_error(x, out)# + tf.losses.get_regularization_loss()
	# loss1 = tf.reduce_mean(tf.square(out - x))
	optimizer = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(alfa,gs,100000,0.1))
	grads = optimizer.compute_gradients(loss)
	optimize = optimizer.minimize(loss, global_step = gs)


	with tf.device('/cpu:0'):
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

			logTestLossSummary = tf.summary.scalar("Test loss log", tf.log(testLoss))
			logTrainLossSummary = tf.summary.scalar("Train loss log", tf.log(trainLoss))

		lossSummaries = tf.summary.merge([testLossSummary, logTestLossSummary, trainLossSummary, logTrainLossSummary])


config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(graph=graph1, config=config)

def displayRec(images):
	asdf = sess.run(out, feed_dict={x :images})
	for i in asdf:
		plt.imshow(i)
		plt.show()

summ_writer = tf.summary.FileWriter("summaries", sess.graph)

# # with tf.Session(graph=graph1) as sess:
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

sess.run(tf.global_variables_initializer(), options=run_opts)

np.random.shuffle(imgs)
img = imgs[:-5]
test = imgs[-5:]
M = img.shape[0]
batchsize = 1
for i in range(10000):
	print("epoch: ", i)
	np.random.shuffle(img)

	for j in range(int(M/batchsize)):
		sess.run(optimize, feed_dict={x: img[j*batchsize:(j+1)*batchsize], alfa:0.00001}, options=run_opts)

	print("train: ", sess.run(loss, feed_dict={x: img}))
	print("test: ", sess.run(loss, feed_dict={x: test}))

	summ1 = sess.run(lossSummaries, feed_dict={ 
							trainLoss: sess.run(loss, feed_dict={x:img}),
							testLoss: sess.run(loss, feed_dict={x:test}) 
							})

	summ2 = sess.run(gSummaries, feed_dict={x: img[j*batchsize:(j+1)*batchsize], alfa:0.001})

	summ_writer.add_summary(summ1, i)
	summ_writer.add_summary(summ2, i)

for i in sess.run(out, feed_dict={x: img[-5:]}):
	plt.imshow(i)
	plt.show()

for i in sess.run(out, feed_dict={x: test}):
	plt.imshow(i)
	plt.show()
