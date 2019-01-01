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
	def __init__(self, trainDB, testDB, trainMeta, testMeta, hparameters=hparametersDef):
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

		self.trainMeta = trainMeta
		self.testMeta = testMeta

		self.createGraph(trainDB, testDB, trainMeta["shape"])

	def createGraph(self, trainDB, testDB, dataShape):
		self.shapes = []

		# Database iterator and operations to reinitialize the iterator
		iter = tf.data.Iterator.from_structure(trainDB.output_types, trainDB.output_shapes)
		self.train_init = iter.make_initializer(trainDB)
		self.test_init = iter.make_initializer(testDB)

		# Input layer
		x = tf.to_float(iter.get_next())
		self.x = x

		# Learning rate
		self.lr = tf.placeholder(tf.float32, shape=(), name="learning_rate")

		# Global learning step counter (Used for learning rate decay calculation)
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

		# Loss and loss optimizer operations
		self.loss = tf.losses.mean_squared_error(x, self.out) + tf.losses.get_regularization_loss()

		self.learning_rate = tf.train.exponential_decay(self.lr, gs, self.alphaTau, 0.1)

		optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
		self.optimize = optimizer.minimize(self.loss, global_step = gs)

		# Operations for creating summaries (Mainly for tensorboard use) 

		# Gradient summaries
		# grads = optimizer.compute_gradients(self.loss)
		# gradSummaries = []
		# for g,v in grads:
		# 	if not g is None:
		# 		with tf.name_scope("grads"):
		# 			gnorm = tf.sqrt(tf.reduce_mean(tf.square(g)))
		# 			gradSummaries.append(tf.summary.scalar(v.name, gnorm))
		# 	else:
		# 		print("Vanishing Grad: ", v.name, g)
		# gSummaries = tf.summary.merge(gradSummaries)

		# Loss summaries
		with tf.name_scope("loss_data"):
			self.testLoss = tf.placeholder(tf.float32, shape=None, name="test_loss_summary")
			self.trainLoss = tf.placeholder(tf.float32, shape=None, name="train_loss_summary")
			testLossSummary = tf.summary.scalar("TestLoss", self.testLoss)
			trainLossSummary = tf.summary.scalar("TrainLoss", self.trainLoss)
			logTestLossSummary = tf.summary.scalar("TestLossLog", tf.log(self.testLoss))
			logTrainLossSummary = tf.summary.scalar("TrainLossLog", tf.log(self.trainLoss))
		lossSummaries = tf.summary.merge([trainLossSummary, testLossSummary, logTrainLossSummary, logTestLossSummary])

		# self.summaries=tf.summary.merge([gSummaries, lossSummaries])
		self.summaries=tf.summary.merge([lossSummaries])
		self.saver = tf.train.Saver()

	def createDenseLayers(self,x):
		N=np.prod(x.get_shape().as_list()[1:])
		with tf.variable_scope("Dense"):
			A = tf.layers.flatten(x)
			self.shapes.append(tf.shape(A))
			A = tf.layers.dense(A, self.latDim, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lam))
			self.feat = A
			A = tf.layers.dropout(A)
			self.shapes.append(tf.shape(A))

			A = tf.layers.dense(A, N, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lam))
			A = tf.layers.dropout(A)
			self.shapes.append(tf.shape(A))

			A = tf.reshape(A, tf.shape(x))
			self.shapes.append(tf.shape(A))
		return A

	def train(self, dirname="summaries",  niter=1000, batchsize=2, display=False, restart=True, printLoss=True):
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
						'filterBase': self.filterBase,
						'latDim': self.latDim
						}

		# Make directory for outputing result info
		if not os.path.exists(dirname):
			os.makedirs(dirname)

		# Recording hyperparamters used for the model
		with open(os.path.join(dirname,"hparameters"), "w") as f:
			f.write(json.dumps(hparameters))

		# Writer for tensorboard information
		summ_writer = tf.summary.FileWriter(dirname)

		with tf.Session(config=config) as sess:

			if restart:
				sess.run(tf.global_variables_initializer())

			# Display network architectue
			sess.run(self.train_init)
			if display:
				print("Architecture:")
				for shape in sess.run(self.shapes):
					print(shape)
				input("Press any key to continue...")

			# Main loop
			for epoch in range(niter):
				print(epoch)

				# Run one epoch of train and calculate average loss on train data
				sess.run(self.train_init)
				finTrain = 0
				for j in range(int(self.trainMeta["length"]/batchsize)):
					_, tmp = sess.run([self.optimize, self.loss], feed_dict={self.lr: self.alpha0})
					finTrain+=tmp
				finTrain/=(self.trainMeta["length"]/batchsize)

				# Calculate loss on test data
				sess.run(self.test_init)
				finTest = 0
				for j in range(self.testMeta["length"]):
					tmp = sess.run(self.loss)
					finTest+=tmp
				finTest/=self.testMeta["length"]

				# Results after 1 epoch of training
				if printLoss:
					print("Train: ", finTrain)
					print("Test: ", finTest)

				# Write summaries for tensorboard
				summ = sess.run(self.summaries, feed_dict={ self.trainLoss: finTrain, self.testLoss: finTest })
				summ_writer.add_summary(summ, epoch)

				# Write results to a file
				with open(os.path.join(dirname,"result"), "a") as f:
					f.write(json.dumps({'finTestLoss': float(finTest), 'finTrainLoss':float(finTrain)}))

			# Save final model parameters
			self.saver.save(sess, os.path.join(dirname,"model"))

			# Displaying original and reconstructed images for visual validation
			if display:
				sess.run(self.test_init)
				for i in range(self.testMeta["length"]):
					x, out = sess.run([self.x, self.out])
					plt.imshow(x[0])
					plt.show()
					plt.imshow(out[0])
					plt.show()

				print("test done")

				sess.run(self.train_init)
				for i in range(int(self.trainMeta["length"]/batchsize)):
					xs, outs = sess.run([self.x, self.out])
					for x, out in zip(xs, outs):
						plt.imshow(x)
						plt.show()
						plt.imshow(out)
						plt.show()

		return finTest, finTrain
