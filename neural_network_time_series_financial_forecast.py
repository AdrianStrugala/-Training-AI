# https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

#przewidywanie kursu gieldowego appla
#siec rekurencyjna

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM #long-short term memory
from keras.layers import Dropout

stock_train_df = pd.read_csv('apple_kurs_train.csv')
stock_train_processed_df = stock_train_df.iloc[:, 1:2].values

# https://arxiv.org/pdf/1510.01378.pdf
scaler = MinMaxScaler(feature_range = (0, 1))
stock_scaled = scaler.fit_transform(stock_train_processed_df)

X = []
y = []
for i in range(20, 755):
    X.append(stock_scaled[i-20:i, 0])
    y.append(stock_scaled[i, 0])
    
X, y = np.array(X), np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(units=20, input_shape=(X.shape[1], 1)))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X, y, epochs = 400, batch_size = 20)


stock_test_df = pd.read_csv('apple_kurs_test.csv')
stock_test_processed_df = stock_test_df.iloc[:, 1:2].values


stock_total = pd.concat((stock_test_df['Open'], stock_train_df['Open']), axis=0)

test_inputs = stock_total[len(stock_total) - len(stock_test_df) - 20:].values


test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)

X_test = []
for i in range(20, len(test_inputs)-1):
    X_test.append(test_inputs[i-20:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(len(X_test))

y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)

plt.figure(figsize=(10,6))
plt.plot(stock_test_processed_df, color='blue', label='Actual Apple Stock Price')
plt.plot(y_pred , color='red', label='Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()