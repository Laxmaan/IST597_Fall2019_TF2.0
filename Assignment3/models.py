from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


class Layer(tf.Module):
   def __init__(self, input_dim, output_dim, name=None):
     super(Layer, self).__init__(name=name)
     self.W = tf.Variable(tf.random.normal([input_dim, output_dim]) , name='W')
     self.b = tf.Variable(tf.zeros([output_dim]), name='b')

   def __call__(self, data):
       print(data.shape, self.W.shape)
     yhat = tf.matmul(data, self.W) + self.b
     return tf.nn.relu(yhat)



class MLP(tf.Module):
    def __init__(self,num_layers,input_size,output_size, name=None,hidden_size=256):
        super(MLP, self).__init__(name=name)
        self.num_layers=num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.layers=[]
        print("here")
        self.hidden_size=hidden_size
        self.layers.append(Layer(input_dim=self.input_size,output_dim=self.hidden_size)) #input layer
        print("here1")
        for i in range(num_layers-1):
            print(i)
            self.layers.append(Layer(input_dim=self.hidden_size,output_dim=self.hidden_size))

        self.layers.append(Layer(input_dim=self.hidden_size,output_dim=self.output_size)) #output layer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def forward_pass(self,data):
        x = data
        for layer in self.layers:
            x=layer(x)
            print(x.shape)
        return x

    def fit(self,X_train,y_train,epochs=50):
        X_train = tf.Variable(x_train,dtype=tf.float32)
        Y_train = tf.one_hot(y_train,10)

        for i in range(epochs):
            print(f"epoch :{i}")
            with tf.GradientTape() as tape:

                logits = self.forward_pass(X_train)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_train)
                loss = tf.reduce_mean(cross_entropy)
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            print(f"Loss = {loss}")
