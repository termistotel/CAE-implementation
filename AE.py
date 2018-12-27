import tensorflow as tf
import numpy as np
import os
import random
import json
import matplotlib.pyplot as plt
from time import sleep

hparametersDef = {"alphaTau": 10000, "alpha0": 0.001, "lam": 0.001, "betaMom": 0.9,
				"betaMom2": 0.999, "convPerLayer": 1, "deconvPerLayer": 1,
				"convLayerNum": 3, "filterNum0": 6, "filterBase": 0, "f": 3,
				"dLayNum": 0, "dLayNeurBase": 0.1, "latDim": 20}

class CAE():
	def __init__(self, dataShape, hparameters=hparametersDef):
		self.alphaTau = hparameters["alphaTau"]
		self.alpha0 = hparameters["alpha0"]
		self.lam = hparameters["lam"]
		self.betaMom = hparameters["betaMom"]
		self.betaMom2 = hparameters["betaMom2"]

		self.f = hparameters["f"]

		self.convPerLayer = hparameters["convPerLayer"]
		self.deconvPerLayer = hparameters["deconvPerLayer"]
		self.convLayerNum = hparameters["convLayerNum"]

		self.dLayNum = hparameters["dLayNum"]
		self.dLayNeurBase = hparameters["dLayNeurBase"]

		self.filterNum0 = hparameters["filterNum0"]
		self.filterBase = hparameters["filterBase"]

		self.filterNum = list(map(lambda x: int(self.filterNum0 * self.filterBase**x), range(self.convLayerNum)))
		self.dLayNeur = list(map(lambda x: self.dLayNeurBase**x, range(self.dLayNum+1)))

		self.latDim = hparameters["latDim"]

		graph = self.createGraph(dataShape)

	def createGraph(self, dataShape):
		self.graph = tf.Graph()
		self.shapes = []
		with self.graph.as_default():
			x = tf.placeholder(shape= ([None]+list(dataShape)), dtype=tf.float32, name="X")
			self.x = x
			self.lr = tf.placeholder(tf.float32, shape=(), name="learning_rate")
			gs = tf.Variable(0, trainable=False)

			# create convolutional layers
			A = x
			self.shapes.append(tf.shape(A))
			convLayersOut = []
			for i in range(self.convLayerNum):
				with tf.variable_scope("Conv"+str(i)):
					for j in range(self.convPerLayer):
						A = tf.layers.conv2d(A, self.filterNum[i], (self.f, self.f), (1,1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lam))
						A = tf.layers.dropout(A)
					if i < (self.convLayerNum-1):
						A = tf.layers.max_pooling2d(A, (2,2), (2,2), padding="SAME")
					self.shapes.append(tf.shape(A))
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
						A = tf.layers.dropout(A)
					self.shapes.append(tf.shape(A))

			# Output layer
			with tf.variable_scope("Upscale"+str(0)):
				A = tf.image.resize_images(A, size=tf.shape(x)[1:3])
				for j in range(self.deconvPerLayer):
					A = tf.layers.conv2d(A, 3, (self.f, self.f), (1,1), padding="SAME", activation=tf.nn.sigmoid, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lam))
					A = tf.layers.dropout(A)
				self.shapes.append(tf.shape(A))
				self.out = A

			self.loss = tf.losses.mean_squared_error(x, self.out) + tf.losses.get_regularization_loss()

			self.learning_rate = tf.train.exponential_decay(self.lr, gs, self.alphaTau, 0.1)
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
					print("Vanishing Grad: ", v.name, g)
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
			A = tf.layers.dense(A, self.latdim, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lam))
			self.feat = A
			A = tf.layers.dropout(A)
			self.shapes.append(tf.shape(A))

			A = tf.layers.dense(A, N, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lam))
			A = tf.layers.dropout(A)
			self.shapes.append(tf.shape(A))

			A = tf.reshape(A, tf.shape(x))
			self.shapes.append(tf.shape(A))
		return A

	def train(self, train, test, dirname="summaries",  niter=1000, batchsize=2, display=False, restart=True, printLoss=True):
		config = tf.ConfigProto(log_device_placement=True)
		# config.gpu_options.allow_growth = True
		hparameters = {
						'niter': niter,
						'batchsize': batchsize,
						'alpha0': self.alpha0,
						'alphaTau': self.alphaTau,
						'lam': self.lam,
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

			if display:
				print("Architecture:")
				for shape in sess.run(self.shapes, feed_dict={self.x:train}):
					print(shape)
				input("Press any key to continue...")

			M = train.shape[0]

			for epoch in range(niter):
				print(epoch)

				# Shuffle train data
				shuffleList = np.arange(M)
				np.random.shuffle(shuffleList)

				train = train[shuffleList]

				for j in range(M//batchsize):
					sess.run(self.optimize, feed_dict={self.x: train[j*batchsize:(j+1)*batchsize], self.lr: self.alpha0})

				summ = sess.run(self.summaries, feed_dict={
										self.trainLoss: sess.run(self.loss, feed_dict={self.x:train}),
										self.testLoss: sess.run(self.loss, feed_dict={self.x:test}),
										self.x: train[j*batchsize:(j+1)*batchsize]
										# self.mask: trainMasks[j*batchsize:(j+1)*batchsize]
										})

				summ_writer.add_summary(summ, epoch)

				finTest = sess.run(self.loss, feed_dict={self.x:test})
				finTrain = sess.run(self.loss, feed_dict={self.x:train})
				if printLoss:
					print("Test: ", finTest)
					print("Train: ", finTrain)

			with open(os.path.join(dirname,"result"), "w") as f:
				f.write(json.dumps({'finTestLoss': float(finTest), 'finTrainLoss':float(finTrain)}))

			self.saver.save(sess, os.path.join(dirname,"model"))

			if display:
				out = sess.run(self.out, feed_dict={self.x: test})
				out2 = sess.run(self.out, feed_dict={self.x: out})
				for i, val in enumerate(out2):
					plt.imshow(test[i])
					plt.show()
					plt.imshow(out[i])
					plt.show()
					plt.imshow(val)
					plt.show()

				print("test done")

				out = sess.run(self.out, feed_dict={self.x: train})
				out2 = sess.run(self.out, feed_dict={self.x: out})
				for i, val in enumerate(out2):
					plt.imshow(train[i])
					plt.show()
					plt.imshow(out[i])
					plt.show()
					plt.imshow(val)
					plt.show()

		return finTest, finTrain
