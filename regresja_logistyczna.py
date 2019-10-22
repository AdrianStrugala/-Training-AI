#Klasyfikacja do 2 stanów

#Wzor : e^(ax+b)/(1-e^(ax+b))

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

x = [[1], [2], [3], [4], [5], [6], [7], [8], [9]]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1]

plt.plot(x, y, 'ro')

#plt.show()

logistic_regression = LogisticRegression()
logistic_regression.fit(x, y)

logistic_regression.coef_      # parametr a
logistic_regression.intercept_ # parametr b 

print(logistic_regression.predict([[3]]))