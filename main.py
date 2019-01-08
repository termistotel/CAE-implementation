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
from sklearn.model_selection import train_test_split

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

def readImage(file, newsize):
  img = cv2.imread(os.path.join(datadir, file))
  img = cv2.resize(img, newsize)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def buildDatabase(datadir, shape):
  print("Rebuiling databases from " + datadir + "...")
  imgs = np.array([readImage(file, shape) for file in sorted(os.listdir(datadir))])
  imgs = imgs/255

  # random.seed(1337)
  shuffleList = np.arange(len(imgs))
  np.random.shuffle(shuffleList)
  imgs = imgs[shuffleList]

  # Split images into train, dev and test set
  train, devtest = train_test_split(imgs, test_size=0.2)
  dev, test = train_test_split(devtest, test_size=0.5)

  print(train.shape, dev.shape, test.shape)
  input()

  # Save train set database
  with open("train.data", 'wb') as data:
    for img in train:
      data.write((img).reshape(-1).tostring())

  # Save dev set database
  with open("dev.data", 'wb') as data:
    for img in dev:
      data.write((img).reshape(-1).tostring())

  # Save test set database
  with open("test.data", 'wb') as data:
    for img in test:
      data.write((img).reshape(-1).tostring())

  # Save train set metadata
  with open("train.metadata", 'w') as mdata:
    mdata.write(json.dumps({'shape': train.shape[1:], 'length': len(train), "dtype": str(train.dtype)}))

  # Save train set metadata
  with open("dev.metadata", 'w') as mdata:
    mdata.write(json.dumps({'shape': dev.shape[1:], 'length': len(dev), "dtype": str(dev.dtype)}))

  # Save test set metadata
  with open("test.metadata", 'w') as mdata:
    mdata.write(json.dumps({'shape': test.shape[1:], 'length': len(test), "dtype": str(test.dtype)}))


def getMetaData():
  with open("train.metadata",'r') as trainf:
    trainm = json.load(trainf)

  with open("dev.metadata",'r') as devf:
    devm = json.load(devf)

  with open("test.metadata",'r') as testf:
    testm = json.load(testf)

  return trainm, devm, testm


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


# Remake databases
if remake:
  imageList = sorted(os.listdir(datadir))
  sizes = map(lambda file: Image.open(os.path.join(datadir, file)).size, imageList)
  minShape = sorted(list(set(sizes)))[1]
  print("Reshaping images to: " + str(minShape))
  input("Press any key to continue...")
  buildDatabase(datadir, minShape)

# Setting up database
if inMem:
  print("Setting up database from memory...")

  imageList = sorted(os.listdir(datadir))
  sizes = map(lambda file: Image.open(os.path.join(datadir, file)).size, imageList)
  minShape = sorted(sizes)[1]

  imgs = np.array([readImage(file, minShape) for file in imageList])
  imgs = imgs/255.0

  # random.seed(1337)
  shuffleList = np.arange(len(imgs))
  np.random.shuffle(shuffleList)
  imgs = imgs[shuffleList]

  # Split images into train, dev and test set
  train, devtest = train_test_split(imgs, test_size=0.2)
  dev, test = train_test_split(devtest, test_size=0.5)

  trainMeta = {"shape": train.shape[1:], "length": len(train), "dtype": str(train.dtype)}
  devMeta = {"shape": dev.shape[1:], "length": len(dev), "dtype": str(dev.dtype)}
  testMeta = {"shape": test.shape[1:], "length": len(test), "dtype": str(test.dtype)}

  trainDB = tf.data.Dataset.from_tensor_slices(train)
  devDB = tf.data.Dataset.from_tensor_slices(dev)
  testDB = tf.data.Dataset.from_tensor_slices(test)

else:
  print("Setting up database from files: " + "train.data, dev.data & test.data")

  trainMeta, devMeta, testMeta = getMetaData()

  bytesPerTrainPoint = np.prod(trainMeta["shape"]) * np.dtype(trainMeta["dtype"]).itemsize
  bytesPerDevPoint = np.prod(devMeta["shape"]) * np.dtype(devMeta["dtype"]).itemsize
  bytesPerTestPoint = np.prod(testMeta["shape"]) * np.dtype(testMeta["dtype"]).itemsize

  trainDB = tf.data.FixedLengthRecordDataset("train.data", bytesPerTrainPoint)
  devDB = tf.data.FixedLengthRecordDataset("dev.data", bytesPerDevPoint)
  testDB = tf.data.FixedLengthRecordDataset("test.data", bytesPerTestPoint)

  trainDB = trainDB.map(lambda x: tf.reshape(tf.decode_raw(x, np.dtype(trainMeta["dtype"])), trainMeta["shape"]))
  devDB = devDB.map(lambda x: tf.reshape(tf.decode_raw(x, np.dtype(devMeta["dtype"])), devMeta["shape"]))
  testDB = testDB.map(lambda x: tf.reshape(tf.decode_raw(x, np.dtype(testMeta["dtype"])), testMeta["shape"]))


trainDB = trainDB.shuffle(100).repeat().batch(batchsize)
devDB = devDB.shuffle(100).repeat().batch(1)
testDB = testDB.shuffle(100).repeat().batch(1)

test, dev, train, imgs = None, None, None, None


hparameters = {}

hparameters = loadHparams("startPoint")

# hparameters["alpha0"] = hparameters["alpha0"]/10
# hparameters["alphaTau"] = 200000
# hparameters["lam"] = 1*1e-6
hparameters["lam"] = 0.0
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
hparameters["latDim"] = 30

hparameters["batchsize"] = batchsize
hparameters["niter"] = niter

print(hparameters)
input("Press any key to continue")

dirname = os.path.join(sumdir, "test")

model = CAE(trainDB, devDB, testDB, trainMeta, devMeta, testMeta,
      dirname=dirname, hparameters = hparameters)
model.train(niter=hparameters["niter"],
      batchsize=hparameters["batchsize"],
      restart=True,
      display=True,
      printLoss=True)

i+=1

