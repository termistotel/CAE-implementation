# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import json
from AE import CAE, hparametersDef

resize = 0.05
npop = 10

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


def clip(x, num):
	if x < num:
		return num
	else:
		return x

def randomInit():
	hparameters = {}

	hparameters["alpha0"] = 0.1**random.gauss(4.0, 0.5)
	hparameters["alphaTau"] = 10**random.gauss(4, 0.2)
	hparameters["lam"] = 0.1**random.gauss(7, 1)

	hparameters["betaMom"] = 1 - 0.1**random.gauss(1,0.4)
	if hparameters["betaMom"] > 0.999:
		hparameters["betaMom"] = 0.999

	hparameters["betaMom2"] = 1 - 0.1**random.gauss(3,0.4)
	hparameters["f"] = int(random.gauss(3.5, 0.4))
	hparameters["f"] = 3

	hparameters["convLayerNum"] = int(random.gauss(2.5, 0.4))
	if hparameters["convLayerNum"]<2:
		hparameters["convLayerNum"]=2
	hparameters["convLayerNum"] = 5

	hparameters["convPerLayer"] = int(random.randint(1, 4))
	hparameters["deconvPerLayer"] = int(random.randint(1, 3))
	hparameters["dLayNum"] = int(random.gauss(1, 0.4))
	hparameters["dLayNum"] = 0

	if hparameters["dLayNum"] < 0:
		hparameters["dLayNum"] = 1

	hparameters["dLayNeurBase"] = random.gauss(0.5, 0.1)
	if hparameters["dLayNeurBase"] < 0.01:
		hparameters["dLayNeurBase"] = 0.01

	hparameters["filterNum0"] = random.randint(8, 16)
	hparameters["filterBase"] = random.gauss(1.8365, 0.4)
	# hparameters["batchsize"] = random.randint(1,3)

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
		return 0.1**random.gauss(4.0, 0.5)

	if name == "alphaTau":
		return 10.0**random.gauss(4.0, 0.2)

	if name == "lam":
		return 0.1**random.gauss(7, 1)

	if name == "betaMom":
		betaMom = 1 - 0.1**random.gauss(1, 0.4)
		if betaMom > 0.999:
			betaMom = 0.999
		return betaMom

	if name == "betaMom2":
		return 1 - 0.1**random.gauss(3, 0.4)

	if name == "f":
		return int(random.gauss(3.5, 0.4))

	if name == "convLayerNum":
		cLN = int(random.gauss(2.5, 0.4))
		# return clip(cLN, 2)
		return 5

	if name == "convPerLayer":
		return int(random.randint(1, 4))

	if name == "deconvPerLayer":
		return int(random.randint(1, 3))

	if name == "dLayNum":
		dLayNum = int(random.gauss(1,0.4))
		return clip(dLayNum, 1)

	if name == "dLayNeurBase":
		return clip(random.gauss(0.5, 0.1), 0.01)

	if name == "filterNum0":
		return random.randint(4, 18)

	if name == "filterBase":
		return clip(random.gauss(1.8365, 0.4), 0.05)

if __name__ == '__main__':
	imgs = np.array([cv2.cvtColor(cv2.resize(cv2.imread("data/"+file), (0,0), fx=resize, fy=resize), cv2.COLOR_BGR2RGB) for file in sorted(os.listdir("data"))])/255
	masks = np.array(list(map(loadMask, sorted(os.listdir("masks")))))

	hparameters = [randomInit() for i in range(npop)]

	generation = 0
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

		fitness = []

		# Evaluating population
		for i, hparameter in enumerate(hparameters):
			# tmp = np.array(list(map(lambda x: x[1], hparameter.items())))
			# fitness.append(1/np.sum(np.square(tmp-test)))
			model = CAE(imgs.shape[1:], hparameter)
			testLoss, trainLoss = model.train(train, test, trainMasks, testMasks,
											dirname=os.path.join(os.path.join("summaries", str(generation)), str(i)),
											niter=200,
											batchsize=1,
											display=False,
											printLoss=False)
			fitness.append((testLoss+trainLoss)/2.0)


		# Selection
		newHparameters = []
		chosen = list(reversed(np.argsort(fitness)))[:5]
		tmp = np.array(list(map(lambda x: x[1], hparameters[chosen[0]].items())))
		print("Best of generation " + str(generation), tmp)
		with open(os.path.join(os.path.join("summaries", str(generation)), "geneticReport"), "w") as f:
			f.write(str(chosen)+"\n"+str(list(reversed(sorted(fitness)))))

		for i, index1 in enumerate(chosen):
			for index2 in chosen[i:]:
				genes1 = hparameters[index1]
				genes2 = hparameters[index2]
				newHparameters.append(crossover(genes1, genes2))

		# Mutation
		for child in newHparameters:
			for key in child:
				r = random.random()
				if r < 0.4:
					child[key] = generateNewGene(key)

		# Add random children
		newHparameters = newHparameters[:npop] + [randomInit() for i in range(npop - len(newHparameters))]

		hparameters = newHparameters
		generation+=1


