import sys
import pandas as pd
import numpy as np
import math
import mykmeanssp as ks
np.random.seed(1234)



# compute the euclidean distance between each data point and the closest center
def compute_dist(data,centers):
  distances = []
  for point in data:
    point = point[1:]
    minDist = math.inf
    for tCenter in centers:
      center = tCenter[0]
      square = np.square(point - center)
      sum_square = np.sum(square)
      dist = np.sqrt(sum_square)
      if (minDist > dist):
        minDist = dist
    distances.append(minDist)
  return distances

# initialize the K centers according to algorithm 1
def init_centers(data):
  data = data.to_numpy()
  centers = []
  # choose one center uniformly at random
  centerIndex = np.random.choice([i for i in range(len(data))])
  center = data[centerIndex][1:]
  index = int(data[centerIndex][0])
  data = np.delete(data, centerIndex, axis=0)
  centers.append((center,index))

  while(len(centers) < K):
    distances = compute_dist(data, centers)
    probabilities = [dist/sum(distances) for dist in distances]
    centerIndex = np.random.choice([i for i in range(len(data))], p=probabilities)
    center = data[centerIndex][1:]
    index = int(data[centerIndex][0])
    data = np.delete(data, centerIndex, axis=0)
    centers.append((center, index))
  return centers

# step 1 Reading user CMD arguments
K = int(sys.argv[1])
eps = float(sys.argv[len(sys.argv)-3])
file_name_1 = sys.argv[len(sys.argv)-2]
file_name_2 = sys.argv[len(sys.argv)-1]
if(len(sys.argv) < 6):
  iter = 300
else:
  iter = int(sys.argv[2])

if(K <= 1):
  print("Invalid number of clusters!")
  exit()
elif(iter <= 1 or iter >= 1000):
  print("Invalid maximum iteration!")
  exit()
elif(eps < 0):
  print("Invalid epsilon!")
  exit()

# step 2+3 combining input files and sorting
df1 = pd.read_csv(file_name_1, sep=",", header=None)
df2 = pd.read_csv(file_name_2, sep=",", header=None)
data = df1.merge(df2,how="inner", on=[0]).sort_values(by=[0])

if(K >= len(data)):
  print("Invalid number of clusters!")
  exit()

# step 4
centersTuple = init_centers(data)
centers = [center for (center, i) in centersTuple]
indices = [i for (center,i) in centersTuple]

d = len(centers[0])
N = len(data)

centersJoined = []
for center in centers:
  for point in center:
    centersJoined.append(point)

dataToList = data.drop(0, axis=1).values.tolist()
dataJoined = []
for lst in dataToList:
  for point in lst:
    dataJoined.append(point)

try:
  centroids = ks.fit(centersJoined, dataJoined, K, N, d, iter)
except:
  print("An Error Has Occurred")

newCentroids = []
centroid = []
for i in range(len(centroids)):
  centroid.append(centroids[i])
  if i % d == d-1:
    newCentroids.append(centroid)
    centroid = []

print(*indices, sep=",")
for i in range(K):
    print(*["%.4f" % num for num in newCentroids[i]], sep=",")
