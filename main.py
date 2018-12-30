# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import json
from AE import CAE, hparametersDef
import argparse
from PIL import Image

import tensorflow as tf


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


def buildDatabase(datadir, shape):
	print("Rebuiling databases from " + datadir + "...")
	imgs = np.array([cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(datadir, file)), shape), cv2.COLOR_BGR2RGB) for file in sorted(os.listdir(datadir))])/255
	# random.seed(1337)
	shuffleList = np.arange(len(imgs))
	np.random.shuffle(shuffleList)
	imgs = imgs[shuffleList]

	train = imgs[:-20]
	test = imgs[-20:]

	with open("train.data", 'wb') as data:
		for img in train:
			data.write((img).reshape(-1).tostring())

	with open("test.data", 'wb') as data:
		for img in test:
			data.write((img).reshape(-1).tostring())

	with open("train.metadata", 'w') as mdata:
		mdata.write(json.dumps({'shape': train.shape[1:], 'length': len(train), "dtype": str(train.dtype)}))

	with open("test.metadata", 'w') as mdata:
		mdata.write(json.dumps({'shape': test.shape[1:], 'length': len(test), "dtype": str(test.dtype)}))


def getMetaData():
	with open("train.metadata",'r') as trainf:
		trainm = json.load(trainf)

	with open("test.metadata",'r') as testf:
		testm = json.load(testf)

	return trainm, testm


p = argparse.ArgumentParser()

p.add_argument("--imgs", required=False, type=str, default="data/", help='input image dir')
p.add_argument("--sums", required=False, type=str, default="summaries", help='summaries dir')
p.add_argument("--rescale", required=False, type=int, default=1, help='resize scale')
p.add_argument("--niter", required=False, type=int, default=1000, help='number of iterations')
p.add_argument("--batch-size", required=False, type=int, default=1, help='batch size')
p.add_argument("--in-memory", action='store_true', help='Keep data in memory')
p.add_argument("--remake", action='store_true', help='Remake database')

args = p.parse_args()

datadir = args.imgs
niter = args.niter
batchsize = args.batch_size
sumdir = args.sums
rescale = args.rescale
inMem = args.in_memory
remake = args.remake

minShape = sorted(map(lambda file: Image.open(os.path.join(datadir, file)).size, sorted(os.listdir(datadir))))[0]

# Remake databases
if remake:
	buildDatabase(datadir, minShape)

# Setting up database
if inMem:
	print("Setting up database from memory...")
	imgs = np.array([cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(datadir, file)), minShape), cv2.COLOR_BGR2RGB) for file in sorted(os.listdir(datadir))])
	# random.seed(1337)
	shuffleList = np.arange(len(imgs))
	np.random.shuffle(shuffleList)
	imgs = imgs[shuffleList]

	train = imgs[:-20]
	test = imgs[-20:]

	trainMeta = {"shape": train.shape[1:], "length": len(train), "dtype": str(train.dtype)}
	testMeta = {"shape": test.shape[1:], "length": len(test), "dtype": str(test.dtype)}

	trainDB = tf.data.Dataset.from_tensor_slices(train)
	testDB = tf.data.Dataset.from_tensor_slices(test)

else:
	print("Setting up database from files: " + "train.data" + ", " + "test.data")

	trainMeta, testMeta = getMetaData()

	trainDB = tf.data.FixedLengthRecordDataset("train.data", np.prod(trainMeta["shape"])*np.dtype(trainMeta["dtype"]).itemsize)
	testDB = tf.data.FixedLengthRecordDataset("test.data", np.prod(testMeta["shape"])*np.dtype(testMeta["dtype"]).itemsize)

	trainDB = trainDB.map(lambda x: tf.reshape(tf.decode_raw(x, np.dtype(trainMeta["dtype"])), trainMeta["shape"]))
	testDB = testDB.map(lambda x: tf.reshape(tf.decode_raw(x, np.dtype(testMeta["dtype"])), testMeta["shape"]))


trainDB = trainDB.shuffle(100).repeat().batch(batchsize)
testDB = testDB.shuffle(100).repeat().batch(1)

batchnum = int(trainMeta["length"]/batchsize)

test, train, imgs = None, None, None



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
hparameters["latDim"] = 20

hparameters["batchsize"] = batchsize
hparameters["niter"] = niter

print(hparameters)
input("Press any key to continue")

model = CAE(trainDB, testDB, trainMeta, testMeta, hparameters)
model.train(dirname=os.path.join(sumdir, "test"),
			niter=hparameters["niter"],
			batchsize=hparameters["batchsize"],
			display=True,
			printLoss=True)

i+=1

