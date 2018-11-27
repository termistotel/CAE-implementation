import os
import matplotlib as plt
import json
from concurrent.futures import ThreadPoolExecutor as PoolExecutor 

def returnString(index):
  with open('summaries/' + str(index) + '/hparameters', 'r') as saveFile:
    for line in saveFile.readlines():
      return json.loads(line)

if __name__=="__main__":
	chosen = [1,22,28,30,33,48,51,66,71,75]
	with PoolExecutor() as executor:
		a = [x for x in executor.map(returnString, chosen)]

	for i, x in enumerate(a):
		print(chosen[i])
		print(x)