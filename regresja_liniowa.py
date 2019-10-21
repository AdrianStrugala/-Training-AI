from sklearn import linear_model as lm
from pandas import DataFrame
import matplotlib.pyplot as plt

data = {'X': [2, 4, 6, 8, 10],
        'y': [2.5, 10, 32, 40, 60]}

data_df = DataFrame(data = data, columns = ['X', 'y'])

print(data_df)

X = data_df.X.values.reshape(len(data_df), 1)
y = data_df.y.values.reshape(len(data_df), 1)

lr = lm.LinearRegression(fit_intercept = True)
model = lr.fit(X,y)

print('b: %.4f, a: %.4f' % (model.intercept_, model.coef_))

y_pred = lr.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred, color='red')

plt.show()