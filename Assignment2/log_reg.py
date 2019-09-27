"""
author:-aam35
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow_datasets as tfds

import utils
from fashion_mnist.utils.mnist_reader import load_mnist
tf.executing_eagerly()

from sklearn.model_selection import train_test_split

random_seed=917326388

tf.random.set_seed(random_seed)
img_shape=(28,28)


# Define paramaters for the model
learning_rate = 7e-3
batch_size = 256
n_epochs = 100
n_train = None
n_test = None
lamda = 0

# Step 1: Read in data
fmnist_folder = './fashion_mnist'
#Create dataset load function [Refer fashion mnist github page for util function]
#Create train,validation,test split
#train, val, test = utils.read_fmnist(fmnist_folder, flatten=True)

# Step 2: Create datasets and iterator
# create training Dataset and batch it
x_train,y_train = load_mnist(os.path.join("fashion_mnist","data","fashion"),"train")
x_train=x_train/255.0

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=random_seed)


print(x_train.shape,y_train.shape)
n_train = x_train.shape[0]
print(x_val.shape,y_val.shape)


# create testing Dataset and batch it
x_test,y_test = load_mnist(os.path.join("fashion_mnist","data","fashion"),"t10k")
x_test=x_test/255.0
print(x_test.shape,y_test.shape)
n_test = x_test.shape[0]
#############################
########## TO DO ############
#############################
#convert data to tf vars
X_train = tf.Variable(x_train,dtype=tf.float32)
X_val   = tf.Variable(x_val,dtype=tf.float32)
X_test  = tf.Variable(x_test,dtype=tf.float32)

Y_train = tf.one_hot(y_train,10)
Y_val   = tf.one_hot(y_val,10)
Y_test  = tf.one_hot(y_test,10)

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
W, b = tf.Variable(tf.random.normal([784,10],stddev=0.01)), tf.Variable(tf.zeros([10]))
#############################
########## TO DO ############
#############################


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
logits = None
#############################
########## TO DO ############
#############################
logits = tf.add(tf.matmul(X_train, W), b)

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
loss = None
#############################
########## TO DO ############
#############################
loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_train)

# Step 6: define optimizer
# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = None
#############################
########## TO DO ############
#############################
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_train, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

#Step 8: train the model for n_epochs times
train_accs=[]
val_accs=[]
for i in range(n_epochs):
    total_loss = 0
    n_batches = n_train//batch_size
    #print(W)
    for batch in range(n_batches):
        indices = np.random.choice(n_train,batch_size)
        X_batch = tf.Variable(x_train[indices],dtype=tf.float32)
        Y_batch = tf.one_hot(y_train[indices],10)
        #Optimize the loss function

        ################################
        ###TO DO#####
        ############

        with tf.GradientTape() as tape:

            #forward pass
            logits = tf.matmul(X_batch, W)+ b
            current_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_batch))+0.5 * lamda * tf.reduce_sum(tf.square(W))

          #evalute the gradient with the respect to the paramters
            dW,db = tape.gradient(current_loss, [ W, b])
            #print(dW)
        W.assign_sub(dW * learning_rate)
        b.assign_sub(db * learning_rate)
        total_loss+=current_loss

    #train accuracy
    logits=tf.matmul(X_train,W)+b
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_train, 1))
    train_accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    train_accs.append((i,train_accuracy.numpy()))

    #val accuracy
    logits=tf.matmul(X_val,W)+b
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_val, 1))
    val_accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    val_accs.append((i,val_accuracy.numpy()))



    total_loss/=n_batches
    print(f"i={i}, loss = {total_loss:3f} acc={train_accuracy:5f} val_acc={val_accuracy:5f}")

i,train=zip(*train_accs)
plt.plot(i,train,label="train acc")
i,val=zip(*val_accs)
plt.plot(i,val,label="val acc")


plt.legend()
plt.show()

#Step 9: Get the Final test accuracy


preds = tf.nn.softmax(tf.matmul(X_test,W)+b)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_test, 1))
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
print(accuracy.numpy())

#Step 10: Helper function to plot images in 3*3 grid
#You can change the function based on your input pipeline

def plot_images(images, y, yhat=None):
    assert len(images) == len(y) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if yhat is None:
            xlabel = "True: {0}".format(y[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(y[i], yhat[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

#Get image from test set
images = x_test[0:9]

# Get the true classes for those images.
y = y_test[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, y=y)
plot_images(images=images, y=y,yhat=tf.argmax(preds, 1).numpy())

#Second plot weights

def plot_weights(w=None):
    # Get the values for the weights from the TensorFlow variable.
    #TO DO ####

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    #TO DO## obtains these value from W
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_clustered_weights(w=None):
    from sklearn.cluster import KMeans

    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(-1,1)
            kmeans = KMeans(n_clusters=10, random_state=random_seed).fit(image)
            # Set the label for the sub-plot.
            ax.set_xlabel("Clustered: {0}".format(i))
            u=kmeans.predict(image).reshape(img_shape)
            # Plot the image.
            ax.imshow(u,  cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

plot_weights(W.numpy())

plot_clustered_weights(W.numpy())
