import numpy as np
import tensorflow as tf
import Network
from tqdm import tqdm
class SGD:
    def __init__(self, learning_rate, train_x, train_y, model: Network.Network) -> None:
        self.learning_rate = learning_rate
        # Expected shape (n, 1, 28*28)
        self.train_x = train_x
        # Expected shape (n, 1, 10)
        self.train_y = train_y 
        # Network.Network  class object
        self.model = model

    def loss(self, x, y):
        # Computes loss function. It supports different batch sizes.
        return (1.0 / len(x)) * tf.reduce_sum(tf.square(self.model.forward(x) - y), axis=[0,2])

    def train(self, batch_size, iter_num):
        # Implements training loop.
        losses = []
        trajectory = []
        n = len(self.train_x)
        for _ in tqdm(range(iter_num)):
            i = np.random.randint(n - batch_size)
            x = self.train_x[i:i + batch_size]
            y = self.train_y[i:i + batch_size]
            with tf.GradientTape() as tape:
                tape.watch(self.model.get_parameters())        
                target = self.loss(x, y)
            losses.append(target)
            grads = tape.gradient(target, self.model.get_parameters())
            trajectory.append(self.apply_gradients(grads, self.model.get_parameters()))
        return losses, trajectory
        
    def apply_gradients(self, gradients, variables):
        # Applies gradients to variables according to SGD iterations.
        # Returns updated parameters.
        new_parameters = []
        for w, grad in zip(variables, gradients):
            new_parameters.append(w - self.learning_rate * grad)
        self.model.set_parameters(new_parameters)
        return self.model.get_parameters()

