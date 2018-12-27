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

resize = 0.05

npop = 10
mutation = 0.1

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


def clip(x, num):
	if x < num:
		return num
	else:
		return x

def randomInit():
	hparameters = {}
	for i in ["alpha0", "alphaTau", "lam", "betaMom", "betaMom2", "f",
			"convLayerNum", "convPerLayer", "deconvPerLayer", "dLayNum",
			"dLayNeurBase", "filterNum0", "filterBase"]:
		hparameters[i] = generateNewGene(i)
	return hparameters

def crossover(g1, g2):
	newg = {}

	for key in g1:
		r = random.random()
		if r < 0.5:
			newg[key] = g1[key]
		else:
			newg[key] = g2[key]

	return newg

def generateNewGene(name, **kwargs):
	if name == "alpha0":
		return 0.1**random.gauss(4.0, 1)

	if name == "alphaTau":
		return 10.0**random.gauss(4.0, 0.5)

	if name == "lam":
		return 0.1**random.randint(2, 10)

	if name == "betaMom":
		betaMom = 1 - 0.1**random.gauss(1, 0.4)
		if betaMom > 0.999:
			betaMom = 0.999
		return betaMom

	if name == "betaMom2":
		return 1 - 0.1**random.gauss(3, 0.4)

	if name == "f":
		# return int(random.gauss(3.5, 0.4))
		return 3

	if name == "convLayerNum":
		cLN = int(random.gauss(2.5, 0.4))
		# return clip(cLN, 2)
		return 5

	if name == "convPerLayer":
		return int(random.randint(1, 4))

	if name == "deconvPerLayer":
		return int(random.randint(1, 4))

	if name == "dLayNum":
		dLayNum = int(random.gauss(1,0.4))
		# return clip(dLayNum, 1)
		return 0

	if name == "dLayNeurBase":
		return clip(random.gauss(0.5, 0.1), 0.01)

	if name == "filterNum0":
		return random.randint(2, 10)

	if name == "filterBase":
		return clip(random.gauss(2, 0.5), 0.05)

if __name__ == '__main__':
	imgs = np.array([cv2.cvtColor(cv2.resize(cv2.imread("data/"+file), (0,0), fx=resize, fy=resize), cv2.COLOR_BGR2RGB) for file in sorted(os.listdir("data"))])/255

	hparameters = [randomInit() for i in range(npop)]

	generation = 0
	while True:
		# random.seed(1337)

		shuffleList = np.arange(len(imgs))
		np.random.shuffle(shuffleList)

		imgs = imgs[shuffleList]

		train = imgs[:-10]

		test = imgs[-10:]

		fitness = []

		# Evaluating population
		for i, hparameter in enumerate(hparameters):
			model = CAE(imgs.shape[1:], hparameter)
			testLoss, trainLoss = model.train(train, test,
											dirname=os.path.join("summaries", str(generation), str(i)),
											niter=niter,
											batchsize=batchsize,
											display=False,
											printLoss=True)
			fitness.append(testLoss)

		# Report
		chosen = list(np.argsort(fitness))[:5]
		with open(os.path.join("summaries", str(generation), "geneticReport"), "w") as f:
			f.write(str(chosen)+"\n"+str(list(sorted(fitness))) + "\n" +
				"Best of generation " + str(generation) + "\n" +
				str(hparameters[np.array(chosen)]))

		# Selection
		newHparameters = []
		for i, index1 in enumerate(chosen):
			for index2 in chosen[i:]:
				genes1 = hparameters[index1]
				genes2 = hparameters[index2]
				newHparameters.append(crossover(genes1, genes2))

		# Mutation
		for child in newHparameters:
			for key in child:
				r = random.random()
				if r < mutation:
					child[key] = generateNewGene(key)

		# Add random children
		newHparameters = newHparameters[:npop] + [randomInit() for i in range(npop - len(newHparameters))]

		hparameters = newHparameters
		generation+=1
