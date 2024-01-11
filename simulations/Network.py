import tensorflow as tf
class Network:
    def __init__(self, layers):
        self.parameters = []
        self.layers = layers

    def build(self):
        for i in range(len(self.layers) - 1):
            # Weight matrix.
            self.parameters.append(tf.random.normal(shape=(self.layers[i+1], self.layers[i]), dtype=tf.float64))
            # Bias.
            self.parameters.append(tf.random.normal(shape=(self.layers[i+1], 1), dtype=tf.float64))

    def get_parameters(self):
        # Parameters getter.
        return self.parameters
    
    def set_parameters(self, new_parameters):
        # Parameters setter.
        self.parameters = new_parameters
    
    def forward(self, x):
        # Expected x shape (batch, 1, 28*28), where batch > 0.
        batch = x.shape[0]
        input_dim = self.layers[0]
        aux = tf.reshape(x, shape=[batch, input_dim, 1])
        for i in range(len(self.layers) - 1):
            aux = tf.linalg.matmul(a=self.parameters[2 * i], b=aux) + self.parameters[2 * i + 1]
            aux = tf.nn.relu(aux)
        aux = tf.nn.sigmoid(aux)
        return tf.reshape(aux, shape=(batch, 1, 10))



