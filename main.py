import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from AE import CAE


if __name__ == '__main__':
	imgs = np.array([cv2.cvtColor(cv2.resize(cv2.imread("data/"+file), (0,0), fx=0.05, fy=0.05), cv2.COLOR_BGR2RGB) for file in os.listdir("data")])/255

	i = 0
	while True:
		# random.seed(1337)
		np.random.shuffle(imgs)
		train = imgs[:-10]
		test = imgs[-10:]

		# alpha0 = 0.1**random.gauss(3.0, 0.5)
		# alphaTau = 10**random.gauss(4, 0.3)
		# lam = 0.1**random.gauss(5, 1)
		# betaMom = 1 - 0.1**random.gauss(1,0.4)
		# if betaMom > 0.999:
		# 	betaMom = 0.999
		# betaMom2 = 1 - 0.1**random.gauss(3,0.4)
		# f = int(random.gauss(3.5, 0.4))
		# convLayerNum = int(random.gauss(2.5, 0.4))
		# if convLayerNum<2:
		# 	convLayerNum=2
		# convPerLayer = int(random.randint(1, 3))
		# deconvPerLayer = int(random.randint(1, 3))
		# dLayNum = int(random.gauss(1, 0.4))
		# # if dLayNum < 0:
		# dLayNum = 1
		# dLayNeurBase = random.gauss(0.5, 0.1)
		# if dLayNeurBase < 0.01:
		# 	dLayNeurBase = 0.01
		# filterNum0 = random.randint(6, 16)
		# filterBase = random.gauss(1.7, 0.3)
		# batchsize = random.randint(1,3)

		alpha0 = 0.000666
		alphaTau = 10003
		# lam = 2.28e-07
		lam = 0.0
		betaMom = 0.9
		betaMom2 = 0.999
		f = 3
		convLayerNum = 4
		convPerLayer = 3
		deconvPerLayer = 1
		dLayNum = 0
		dLayNeurBase = 0.388
		filterNum0 = 12
		filterBase = 1.8365
		niter = 20
		batchsize = 1

		model = CAE(imgs.shape[1:], alpha0=alpha0, alphaTau=alphaTau, lam=lam, betaMom=betaMom, betaMom2=betaMom2,
					f=f, convLayerNum=convLayerNum, convPerLayer=convPerLayer, dLayNum=dLayNum, dLayNeurBase=dLayNeurBase,
					filterNum0=filterNum0, filterBase=filterBase)
		model.train(train, test, dirname=os.path.join("summaries", str(i)), niter=niter, batchsize=batchsize, display=False)
		i+=1
