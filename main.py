# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import json
from AE import CAE, hparametersDef

niter = 1000
batchsize = 1
sumdir = "summaries"
datadir = "data"

resize = 0.04

def loadHparams(dir):	
	with open(os.path.join(dir,"hparameters"), "r") as f:
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

# if __name__ == '__main__':
imgs = np.array([cv2.cvtColor(cv2.resize(cv2.imread("data/"+file), (0,0), fx=resize, fy=resize), cv2.COLOR_BGR2RGB) for file in sorted(os.listdir("data"))])/255

i = 0
# random.seed(1337)

shuffleList = np.arange(len(imgs))
np.random.shuffle(shuffleList)

imgs = imgs[shuffleList]

train = imgs[:-20]
test = imgs[-20:]

hparameters = {}

hparameters = loadHparams(os.path.join(sumdir, str(21), "21_0"))

# hparameters["alpha0"] = hparameters["alpha0"]/10
# hparameters["alphaTau"] = 200000
# hparameters["lam"] = 1*1e-4
# hparameters["betaMom"] = 0.9
# hparameters["betaMom2"] = 0.999
# hparameters["f"] = 3
# hparameters["convLayerNum"] = 7
# # hparameters["convPerLayer"] = 2
# hparameters["deconvPerLayer"] = 1
# hparameters["dLayNum"] = 1
# hparameters["dLayNeurBase"] = 0.5
# hparameters["filterNum0"] = 8
# hparameters["filterBase"] = 1.8365

hparameters["batchsize"] = batchsize
hparameters["niter"] = niter

print(train.shape)
print(hparameters)
input("Press any key to continue")

model = CAE(imgs.shape[1:], hparameters)
model.train(train, test,
			dirname=os.path.join(sumdir, "test"),
			niter=hparameters["niter"],
			batchsize=hparameters["batchsize"],
			display=True,
			printLoss=True)

i+=1
