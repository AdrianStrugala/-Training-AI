import tensorflow as tf
import numpy as np

np.set_printoptions(suppress=True)

print(tf.__version__)

#dane
X = [
    [0, 0],
    [1, 1],
    [1, 0],
    [0, 1]
]

y = [[0], [0], [1], [1]]

#ksztalt wejscia i wyjsica
input = tf.placeholder(tf.float32, shape=[None, 2])
target = tf.placeholder(tf.float32, shape=[None, 1])

#macierze wag
W1 = tf.Variable(tf.random_uniform([2, 2]))
W2 = tf.Variable(tf.random_uniform([2, 1]))

#bias
b1 = tf.Variable(tf.zeros([2]))
b2 = tf.Variable(tf.zeros([1]))

#wyjscia warstw
a1 = tf.sigmoid(tf.matmul(input, W1) + b1)
a2 = tf.sigmoid(tf.matmul(a1, W2) + b2)

#funkcja wstecznej propagacji
cost = -tf.reduce_sum(target * tf.log(a2) + (1-target) * tf.log(1 - a2))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

epochs = 2000

for i in range(epochs):
    error = sess.run(train_step, feed_dict= {input: X, target: y})
    if i % 100 == 0:
        print('Epoch: ' +str(i) + ', cost: ' + str(sess.run(cost, feed_dict = {input: X, target: y})))


y_pred = sess.run(a2, feed_dict= {input: X, target: y})

print(y_pred)