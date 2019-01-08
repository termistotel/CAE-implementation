# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import json
import tensorflow as tf
from AE import CAE, hparametersDef
import argparse
from sklearn.model_selection import train_test_split

defMus = {"alpha0": 4.0, "alphaTau": 4.0, "lam": 6, "betaMom": 1, "betaMom2": 3, "f": 3,
    "convLayerNum": 2.5, "convPerLayer": 2, "deconvPerLayer": 1, "dLayNum": 1,
    "dLayNeurBase": 0.5, "filterNum0": 6, "filterBase": 2}

defSigmas = {"alpha0": 2, "alphaTau": 0.5, "lam": 4, "betaMom": 0.4, "betaMom2": 0.4, "f": 0.4,
    "convLayerNum": 0.4, "convPerLayer": 1, "deconvPerLayer": 1, "dLayNum": 0.4,
    "dLayNeurBase": 0.1, "filterNum0": 4, "filterBase": 0.5}

keys = ["alpha0", "alphaTau", "lam", "betaMom", "betaMom2", "f",
    "convLayerNum", "convPerLayer", "deconvPerLayer", "dLayNum",
    "dLayNeurBase", "filterNum0", "filterBase", "latDim"]

mutables = ["alpha0", "alphaTau", "lam", "betaMom", "betaMom2", "f",
    "convLayerNum", "convPerLayer", "deconvPerLayer", "dLayNum",
    "dLayNeurBase", "filterNum0", "filterBase", "latDim"]

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


def clip(x, num):
  if x < num:
    return num
  else:
    return x

def randomInit(mus = defMus):
  hparameters = {}
  for i in keys:
    hparameters[i] = generateNewGene(i, mus=defMus)
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

def argForRNG(keyword, mu, sigma, type="gaus"):
  if type == "gaus":
    return mu[keyword], sigma[keyword]
  else:
    return int(mu[keyword] - sigma[keyword]), int(mu[keyword] + sigma[keyword])

def generateNewGene(name, mus=defMus, sigmas=defSigmas):
  if name == "alpha0":
    return 0.1**random.gauss(*argForRNG("alpha0", mus, sigmas, type="gaus"))

  if name == "alphaTau":
    return 10.0**random.gauss(*argForRNG("alphaTau", mus, sigmas, type="gaus"))

  if name == "lam":
    # return 0.1**random.randint(*argForRNG("lam", mus, sigmas, type="uniform"))
    return 0.0

  if name == "betaMom":
    betaMom = clip(1 - 0.1**random.gauss(*argForRNG("betaMom", mus, sigmas, type="gaus")), 0)
    if betaMom > 0.999:
      betaMom = 0.999
    return betaMom

  if name == "betaMom2":
    return clip(1 - 0.1**random.gauss(*argForRNG("betaMom2", mus, sigmas, type="gaus")), 0.01)

  if name == "f":
    # return clip(int(random.gauss(*argForRNG("f", mus, sigmas, type="gaus"))), 1)
    return 3

  if name == "convLayerNum":
    cLN = clip(int(random.gauss(*argForRNG("convLayerNum", mus, sigmas, type="gaus"))), 1)
    # return clip(cLN, 2)
    # return 5
    return random.randint(4,6)

  if name == "convPerLayer":
    return clip(int(random.randint(*argForRNG("convPerLayer", mus, sigmas, type="uniform"))), 1)

  if name == "deconvPerLayer":
    return clip(int(random.randint(*argForRNG("deconvPerLayer", mus, sigmas, type="uniform"))), 1)

  if name == "dLayNum":
    dLayNum = clip(int(random.gauss(*argForRNG("dLayNum", mus, sigmas, type="gaus"))), 0)
    # return clip(dLayNum, 1)
    return 0

  if name == "dLayNeurBase":
    return clip(random.gauss(*argForRNG("dLayNeurBase", mus, sigmas, type="gaus")), 0.01)

  if name == "filterNum0":
    return clip(random.randint(*argForRNG("filterNum0", mus, sigmas, type="uniform")), 1)

  if name == "filterBase":
    return clip(random.gauss(*argForRNG("filterBase", mus, sigmas, type="gaus")), 0.05)

  if name == "latDim":
    return 30

def createDatabase():
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

  trainDB = trainDB.map(augment).shuffle(100).repeat().batch(batchsize)
  devDB = devDB.map(augment).shuffle(100).repeat().batch(1)
  testDB = testDB.map(augment).shuffle(100).repeat().batch(1)

  return trainDB, devDB, testDB, trainMeta, devMeta, testMeta

if __name__ == '__main__':

  p = argparse.ArgumentParser()

  p.add_argument("--imgs", required=False, type=str, default="data/", help='input image dir')
  p.add_argument("--sums", required=False, type=str, default="summaries", help='summaries dir')
  p.add_argument("--rescale", required=False, type=int, default=1, help='resize scale')
  p.add_argument("--niter", required=False, type=int, default=1000, help='number of iterations')
  p.add_argument("--batch-size", required=False, type=int, default=1, help='batch size')
  p.add_argument("--in-memory", action='store_true', help='Keep data in memory')
  p.add_argument("--remake", action='store_true', help='Remake database')
  p.add_argument("--npop",required=False, type=int, default=10, help='Population number per generation')
  p.add_argument("--mchance",required=False, type=float, default=0.1, help='Mutation chance')
  p.add_argument("--colordiv", required=False, type=float, default=0.2, help='Color divergence standard deviation')
  args = p.parse_args()

  colorDiv = args.colordiv
  datadir = args.imgs
  niter = args.niter
  batchsize = args.batch_size
  sumdir = args.sums
  rescale = args.rescale
  inMem = args.in_memory
  remake = args.remake

  npop = args.npop
  mutation = args.mchance

  def augment(img):
    outImg = img
    # 50% chance to flip image in either direction
    outImg = tf.cond(tf.random.uniform([]) < 0.5, lambda: tf.reverse(outImg, axis=[0]) , lambda: outImg)
    outImg = tf.cond(tf.random.uniform([]) < 0.5, lambda: tf.reverse(outImg, axis=[1]) , lambda: outImg)

    # Color Divergance shift
    a = tf.random.normal([1,1,3], mean=1, stddev=colorDiv, dtype=tf.float64)
    outImg = tf.clip_by_value(outImg*a, 0, 1)
    return outImg

  # Remake databases
  if remake:
    imageList = sorted(os.listdir(datadir))
    sizes = map(lambda file: Image.open(os.path.join(datadir, file)).size, imageList)
    minShape = sorted(list(set(sizes)))[1]
    print("Reshaping images to: " + str(minShape))
    input("Press any key to continue...")
    buildDatabase(datadir, minShape)


  def start(startPoint):
    if startPoint is not None:
      loadHparameter = loadHparams(startPoint)
      loadHparameter["latDim"] = generateNewGene("latDim")
      hparameters = [randomInit() for i in range(npop)]
      for i in range(3):
        for key in hparameters[i]:
          hparameters[i][key] = loadHparameter[key]
    else:
      hparameters = [randomInit() for i in range(npop)]
    generation = 0

    return hparameters, generation

  def continueGen(genNum):
    pops = list(os.walk(os.path.join("summaries", str(genNum))))[0][1]
    hparameters = []
    for pop in pops:
      loadHparameter = loadHparams(os.path.join("summaries", str(genNum), pop))
      tmp = {}
      for key in keys:
        tmp[key] = loadHparameter[key]
      hparameters.append(tmp)

    for i in range(npop - len(hparameters)):
      hparameters.append(randomInit())

    return hparameters, genNum+1

  # hparameters, generation = start("startPoint")
  hparameters, generation = continueGen(29)

  print(generation)
  # print(hparameters)
  # input()

  while True:
    fitness = []

    # Evaluating population
    for i, hparameter in enumerate(hparameters):
      trainDB, devDB, testDB, trainMeta, devMeta, testMeta = createDatabase()
      print(i)
      print(hparameter)
      dirname = os.path.join("summaries", str(generation), str(i))
      model = CAE(trainDB, devDB, testDB, trainMeta, devMeta, testMeta,
          dirname=dirname, hparameters=hparameter)
      devLoss, trainLoss = model.train(niter=niter,
                      batchsize=batchsize,
                      display=False,
                      restart=True,
                      printLoss=True)
      tf.reset_default_graph()
      fitness.append(trainLoss)

    # Report
    chosen = list(np.argsort(fitness))[:5]
    with open(os.path.join("summaries", str(generation), "geneticReport"), "w") as f:
      f.write(str(chosen)+"\n"+str(list(sorted(fitness))) + "\n" +
        "Best of generation " + str(generation) + "\n")
      for i in chosen:
        f.write(str(hparameters[i]) + "\n")

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
        if key in mutables:
          r = random.random()
          if r < mutation:
            child[key] = generateNewGene(key)

    # Add random children
    newHparameters = newHparameters[:npop] + [randomInit() for i in range(npop - len(newHparameters))]

    hparameters = newHparameters
    generation+=1
