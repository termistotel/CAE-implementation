import tensorflow as tf
import numpy as np
import os
import random
import json
from kron_layers import kfcPic, kfc
import matplotlib.pyplot as plt

# def displayRec(images):
# 	asdf = sess.run(out, feed_dict={x :images})
# 	for i in asdf:
# 		plt.imshow(i)
# 		plt.show()

class CAE():
	def __init__(self, dataShape, alphaTau=10000, alpha0=0.001, lam=0.001, betaMom=0.9, betaMom2=0.999, convPerLayer=1, deconvPerLayer=1, convLayerNum=3, filterNum0=6, filterBase=0, f=3, dLayNum=0, dLayNeurBase=0.1):
		self.alphaTau = alphaTau
		self.alpha0 = alpha0
		self.lam = lam
		self.betaMom = betaMom
		self.betaMom2 = betaMom2

		self.f = f

		self.convPerLayer = convPerLayer
		self.deconvPerLayer = deconvPerLayer
		self.convLayerNum = convLayerNum

		self.dLayNum = dLayNum
		self.dLayNeurBase = dLayNeurBase

		self.filterNum0 = filterNum0
		self.filterBase = filterBase

		self.filterNum = list(map(lambda x: int(filterNum0 * filterBase**x), range(convLayerNum)))
		self.dLayNeur = list(map(lambda x: dLayNeurBase**x, range(dLayNum+1)))

		graph = self.createGraph(dataShape)

	def createGraph(self, dataShape):
		self.graph = tf.Graph()
		with self.graph.as_default():
			x = tf.placeholder(shape= ([None]+list(dataShape)), dtype=tf.float32, name="X")
			self.x = x
			gs = tf.Variable(0, trainable=False)

			# create convolutional layers
			A = x
			convLayersOut = []
			for i in range(self.convLayerNum):
				with tf.variable_scope("Conv"+str(i)):
					for j in range(self.convPerLayer):
						A = tf.layers.conv2d(A, self.filterNum[i], (self.f, self.f), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lam))
					A = tf.layers.max_pooling2d(A, (2,2), (2,2), padding="SAME")
					convLayersOut.append(A)

			# create dense layers
			A = self.createDenseLayers(A)

			# create upscale layers
			convLayersOut.pop()
			for i in list(reversed(range(self.convLayerNum)))[1:]:
				with tf.variable_scope("Upscale"+str(i+1)):
					A = tf.image.resize_images(A, size=tf.shape(convLayersOut.pop())[1:3])
					for j in range(self.deconvPerLayer):
						A = tf.layers.conv2d(A, self.filterNum[i], (self.f, self.f), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lam))

			# Output layer
			with tf.variable_scope("Upscale"+str(0)):
				A = tf.image.resize_images(A, size=tf.shape(x)[1:3])
				for j in range(self.deconvPerLayer):
					self.out = tf.layers.conv2d(A, 3, (self.f, self.f), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lam))

			self.learning_rate = tf.train.exponential_decay(self.alpha0,gs, self.alphaTau, 0.1)
			self.loss = tf.losses.mean_squared_error(x, self.out) + tf.losses.get_regularization_loss()
			optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
			self.optimize = optimizer.minimize(self.loss, global_step = gs)
			grads = optimizer.compute_gradients(self.loss)

			gradSummaries = []
			for g,v in grads:
				if not g is None:
					with tf.name_scope("grads"):
						gnorm = tf.sqrt(tf.reduce_mean(tf.square(g)))
						gradSummaries.append(tf.summary.scalar(v.name, gnorm))
				else:
					print("eto: ", v.name, g)
			gSummaries = tf.summary.merge(gradSummaries)

			with tf.name_scope("loss_data"):
				self.testLoss = tf.placeholder(tf.float32, shape=None, name="test_loss_summary")
				self.trainLoss = tf.placeholder(tf.float32, shape=None, name="train_loss_summary")
				testLossSummary = tf.summary.scalar("TestLoss", self.testLoss)
				trainLossSummary = tf.summary.scalar("TrainLoss", self.trainLoss)
				logTestLossSummary = tf.summary.scalar("TestLossLog", tf.log(self.testLoss))
				logTrainLossSummary = tf.summary.scalar("TrainLossLog", tf.log(self.trainLoss))
			lossSummaries = tf.summary.merge([testLossSummary, logTestLossSummary, trainLossSummary, logTrainLossSummary])

			self.summaries=tf.summary.merge([gSummaries, lossSummaries])
			self.saver = tf.train.Saver()

		return self.graph

	def createDenseLayers(self,x):
		N=np.prod(x.get_shape().as_list()[1:])
		with tf.variable_scope("Dense"):
			A = tf.layers.flatten(x)
			for i in range(self.dLayNum):
				A = tf.layers.dense(A, N*self.dLayNeur[i+1], activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lam))

			for i in reversed(range(self.dLayNum)):
				A = tf.layers.dense(A, N*self.dLayNeur[i], activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lam))
			A = tf.reshape(A, tf.shape(x))
		return A

	def train(self, train, test, dirname="summaries",  niter=1000, batchsize=2, display=False, restart=True):
		config = tf.ConfigProto(log_device_placement=True)
		# config.gpu_options.allow_growth = True
		hparameters = {
						'niter': niter,
						'batchsiize': batchsize,
						'alpha0': self.alpha0,
						'alphaTau': self.alphaTau,
						'lambda': self.lam,
						'betaMom': self.betaMom,
						'betaMom2': self.betaMom2,
						'fshape': (self.f, self.f),
						'convLayerNum': self.convLayerNum,
						'convPerLayer': self.convPerLayer,
						'deconvPerLayer': self.deconvPerLayer,
						'dLayNum': self.dLayNum,
						'dLayNeurBase': self.dLayNeurBase,
						'filterNum0': self.filterNum0,
						'filterBase': self.filterBase
						}
		if not os.path.exists(dirname):
			os.makedirs(dirname)

		with open(os.path.join(dirname,"hparameters"), "w") as f:
			f.write(json.dumps(hparameters))

		summ_writer = tf.summary.FileWriter(dirname, self.graph)

		with tf.Session(graph=self.graph, config=config) as sess:

			if restart:
				sess.run(tf.global_variables_initializer())

			M = train.shape[0]

			for epoch in range(niter):
				print(epoch)
				np.random.shuffle(train)
				for j in range(M//batchsize):
					sess.run(self.optimize, feed_dict={self.x: train[j*batchsize:(j+1)*batchsize]})

				summ = sess.run(self.summaries, feed_dict={
										self.trainLoss: sess.run(self.loss, feed_dict={self.x:train}),
										self.testLoss: sess.run(self.loss, feed_dict={self.x:test}),
										self.x: train[j*batchsize:(j+1)*batchsize]
										})

				summ_writer.add_summary(summ, epoch)

				finTest = sess.run(self.loss, feed_dict={self.x:test})
				finTrain = sess.run(self.loss, feed_dict={self.x:train})
				print(finTest)
				print(finTrain)

			with open(os.path.join(dirname,"result"), "w") as f:
				f.write(json.dumps({'finTestLoss': float(finTest), 'finTrainLoss':float(finTrain)}))

			self.saver.save(sess, os.path.join(dirname,"model"))

			if display:
				for i, val in enumerate(sess.run(self.out, feed_dict={self.x: test})):
					plt.imshow(test[i])
					plt.show()
					plt.imshow(val)
					plt.show()

				print("test done")

				for i, val in enumerate(sess.run(self.out, feed_dict={self.x: train})):
					plt.imshow(train[i])
					plt.show()
					plt.imshow(val)
					plt.show()
