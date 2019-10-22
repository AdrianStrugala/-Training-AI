#klasyfikacja do 2 stanow

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

X = np.array([
    [1,2],
    [5,8],
    [1.5,1.8],
    [8,8],
    [1,0.6],
    [9,11]
])

y = [0,1,0,1,0,1]

classifier = svm.SVC(kernel = 'linear', C=1.0)
classifier.fit(X, y)

w=classifier.coef_[0]
a = -w[0]/w[1]

xx = np.linspace(0,12)
yy = a * xx - classifier.intercept_[0]/w[1]

h0 = plt.plot(xx,yy,'k--')

plt.scatter(X[:,0], X[:,1], c=y)

plt.show()

print(classifier.predict([(3,2)]))