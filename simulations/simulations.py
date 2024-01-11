import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from keras.datasets import mnist
import keras
from keras import layers
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from SGD import SGD

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

array = [tf.keras.layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dense(10,)]

nn = keras.Sequential(array)

# Preprocessing data.
imgs = tf.reshape(x_train, (len(x_train),28,28))
labels = np.zeros((len(y_train),1,10))
for z, y in zip(labels, y_train):
    z[0][y] = 1

n = len(imgs)
batch = 64
l = []
trajectory = []
optimizer = SGD(0.01)
for _ in tqdm(range(5000)):
    i = np.random.randint(n - batch)
    x = tf.reshape(imgs[i:i + batch], (batch, 28,28))
    with tf.GradientTape() as tape:
        tape.watch(nn.trainable_variables)        
        target = tf.reduce_mean(tf.square(labels[i: i + batch] - nn(x)))
    #List of gradients
    grads = tape.gradient(target, nn.trainable_variables)
    optimizer.apply_gradients(grads, nn.trainable_variables)
    l.append(target)
    trajectory.append(nn.trainable_variables)

plt.plot(l)
plt.grid()
plt.show()