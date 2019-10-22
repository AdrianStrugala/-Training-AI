#liczba klastrow powinna byc ustalana przez punkt przegiecia funkcji (punkt lokciowy)

from numpy import array, vstack
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq

data = vstack((rand(150, 2) + array([.5, .5]), rand(150, 2)))

print(data)

plt.scatter(data[:,0], data[:,1])
#plt.show()

#obliczanie centroidow
centroids, _ = kmeans(data, 5)

plt.scatter(centroids[:,0],centroids[:,1],  color='red')
plt.show()

idx, _ = vq(data, centroids)
