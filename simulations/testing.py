import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from Network import Network
from SGD import SGD
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0 , x_test / 255.0

# Preprocessing data.
n = len(x_train)
x_ds = tf.constant(x_train, shape=(n, 1, 28*28), dtype=tf.float64)

y_ds = np.zeros((n,1,10))
for z, y in zip(y_ds, y_train):
    z[0][y] = 1

# Testing model.
nn = Network([28*28, 128, 128, 128, 10])
nn.build()
sgd = SGD(learning_rate=2, train_x=x_ds, train_y=y_ds, model=nn)

losses = sgd.train(batch_size=64, iter_num=1000)
plt.plot(losses)
#plt.yscale('log')
plt.grid()
plt.show()

#print([np.argmax(nn.forward(x)) for x in x_ds] == [np.argmax(label) for label in labels])