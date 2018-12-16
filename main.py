# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import json
from AE import CAE, hparametersDef

resize = 0.05

def loadMask(file):
	mask = cv2.imread("masks/"+file, cv2.IMREAD_GRAYSCALE)[:1538]
	# mask[mask>(255//2)] = 255
	# mask[mask>(255//2)] = 0
	mask = cv2.resize(mask[1:-1, 1:-1], (0,0), fx=resize, fy=resize)
	# mask[mask>0] = 1
	return mask

def loadHparams(num):	
	with open(os.path.join(os.path.join("summaries", str(num)),"hparameters"), "r") as f:
		hparams = json.load(f)

	# Fixes
	if "batchsiize" in hparams:
		batchsize = hparams["batchsiize"]
		del hparams["batchsiize"]
		hparams["batchsize"] = batchsize

	if "fshape" in hparams:
		f = hparams["fshape"][0]
		del hparams["fshape"]
		hparams["f"] = f

	if "lambda" in hparams:
		lam = hparams["lambda"]
		del hparams["lambda"]
		hparams["lam"] = lam

	return hparams

if __name__ == '__main__':
	imgs = np.array([cv2.cvtColor(cv2.resize(cv2.imread("data/"+file), (0,0), fx=resize, fy=resize), cv2.COLOR_BGR2RGB) for file in sorted(os.listdir("data"))])/255
	masks = np.array(list(map(loadMask, sorted(os.listdir("masks")))))
		
	# for i, val in enumerate(imgs):
	# 	plt.imshow(val)
	# 	plt.show()
	# 	plt.imshow(masks[i])
	# 	plt.show()

	i = 0
	while True:
		# random.seed(1337)

		shuffleList = np.arange(len(imgs))
		np.random.shuffle(shuffleList)

		imgs = imgs[shuffleList]
		masks = masks[shuffleList]

		train = imgs[:-10]
		trainMasks = masks[:-10]

		test = imgs[-10:]
		testMasks = masks[-10:]

		hparameters = {}

		# hparameters = loadHparams(49)

		hparameters["alpha0"] = 0.1**random.gauss(4.0, 0.3)
		hparameters["alphaTau"] = 10**random.gauss(4, 0.2)
		hparameters["lam"] = 0.1**random.gauss(7, 1)

		hparameters["betaMom"] = 1 - 0.1**random.gauss(1,0.4)
		if hparameters["betaMom"] > 0.999:
			hparameters["betaMom"] = 0.999

		hparameters["betaMom2"] = 1 - 0.1**random.gauss(3,0.4)
		hparameters["f"] = int(random.gauss(3.5, 0.4))

		hparameters["convLayerNum"] = int(random.gauss(2.5, 0.4))
		if hparameters["convLayerNum"]<2:
			hparameters["convLayerNum"]=2

		hparameters["convPerLayer"] = int(random.randint(1, 4))
		# hparameters["deconvPerLayer"] = int(random.randint(1, 3))
		hparameters["dLayNum"] = int(random.gauss(1, 0.4))

		if hparameters["dLayNum"] < 0:
			hparameters["dLayNum"] = 1

		hparameters["dLayNeurBase"] = random.gauss(0.5, 0.1)
		if hparameters["dLayNeurBase"] < 0.01:
			hparameters["dLayNeurBase"] = 0.01

		hparameters["filterNum0"] = random.randint(8, 16)
		hparameters["filterBase"] = random.gauss(1.8365, 0.4)
		# hparameters["batchsize"] = random.randint(1,3)


		# hparameters["alpha0"] = hparameters["alpha0"]
		hparameters["alphaTau"] = 20000
		hparameters["lam"] = 0.0
		hparameters["betaMom"] = 0.9
		hparameters["betaMom2"] = 0.999
		hparameters["f"] = 3
		hparameters["convLayerNum"] = 5
		# hparameters["convPerLayer"] = 2
		hparameters["deconvPerLayer"] = 1
		hparameters["dLayNum"] = 0
		# hparameters["dLayNeurBase"] = 0.0
		# hparameters["filterNum0"] = 8
		# hparameters["filterBase"] = 1.8365
		hparameters["batchsize"] = 1

		hparameters["niter"] = 400

		print(train.shape)
		print(hparameters)
		# hparameters["alphaTau"] = 20000
		# model = CAE(imgs.shape[1:], alpha0=alpha0, alphaTau=alphaTau, lam=lam, betaMom=betaMom, betaMom2=betaMom2,
		# 			f=f, convLayerNum=convLayerNum, convPerLayer=convPerLayer, dLayNum=dLayNum, dLayNeurBase=dLayNeurBase,
		# 			filterNum0=filterNum0, filterBase=filterBase)

		model = CAE(imgs.shape[1:], hparameters)
		model.train(train, test, trainMasks, testMasks,
					dirname=os.path.join("summaries", str(i)),
					niter=hparameters["niter"],
					batchsize=hparameters["batchsize"],
					display=False)
		i+=1
