# -*- coding: utf-8 -*-
<<<<<<< HEAD
"""IST assignment 3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EolES9WQNOFzhDcW8pGVv162SLAzZ3tK
"""

# Commented out IPython magic to ensure Python compatibility.
from __future__ import absolute_import, division, print_function, unicode_literals
=======
"""
Author:-aam35
Analyzing Forgetting in neural networks
"""
>>>>>>> master

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn
# %matplotlib inline

random_seed=917326388
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

task_permutation = []
num_tasks_to_run = 10


num_epochs_per_task = 20

minibatch_size = 32
learning_rate = 0.001

for task in range(num_tasks_to_run):
	task_permutation.append( np.random.permutation(784) )

"""#Loading MNIST"""

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train =   x_train.reshape(-1,784)
x_test =   x_test.reshape(-1,784)


train_shape=x_train.shape
test_shape =x_test.shape
num_train = x_train.shape[0]
num_test = x_test.shape[0]
print(x_train.shape)

x_train_1 = x_train[:,task_permutation[0]]
plt.imshow(x_train_1[0].reshape(28,28))


data_index=25
f, axarr = plt.subplots(2,5)
for i in range(2):
  for j in range(5):
    index=5*i+j
    axarr[i,j].imshow(x_train[data_index,task_permutation[index]].reshape(28,28))
plt.show()
"""#Defining the Model"""

class Layer(tf.Module):
   def __init__(self, input_dim, output_dim, name=None,activation="linear",rate=0):
     super(Layer, self).__init__(name=name)
     w_init = tf.random_normal_initializer()
     b_init = tf.zeros_initializer()
     self.activations={"relu":tf.nn.relu,"sigmoid":tf.nn.sigmoid,"softmax":tf.nn.softmax,"linear":lambda x: x}
     self.activation=activation
     self.W = tf.Variable(initial_value=w_init(shape=(input_dim, output_dim),dtype='float32'),
                         trainable=True, name=f'W')
     self.b = tf.Variable(initial_value=b_init(shape=(output_dim,),dtype='float32'),
                         trainable=True, name=f'b')
     self.rate=rate

   def __call__(self, data, training=None):
     if training:
        data= tf.nn.dropout(data, rate=self.rate)
     yhat = tf.matmul(data, self.W) + self.b
     act_fn=self.activations.get(self.activation,lambda x: x)
     return act_fn(yhat)



class MLP(tf.Module):
    def __init__(self,num_layers,input_size,output_size, name=None,hidden_size=256,
                 optimizer=None, print_every=10,dropout_rate=0,batch_size=1024,
                 loss='categorical_crossentropy'):
        super(MLP, self).__init__(name=name)
        self.num_layers=num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.layers=[]
        self.hidden_size=hidden_size
        self.dropout_rate = dropout_rate
        self.batch_size=batch_size
        self.to_one_hot = (loss=='categorical_crossentropy')

        #Declare layers
        self.layers.append(Layer(input_dim=self.input_size,output_dim=self.hidden_size,name='input',activation='relu',rate=self.dropout_rate)) #input layer
        for i in range(1,num_layers):
            self.layers.append(Layer(input_dim=self.hidden_size,output_dim=self.hidden_size,name=f'hidden_{i}',activation="relu",rate=self.dropout_rate))

        output_activation = 'softmax' if self.to_one_hot else 'linear'

        self.layers.append(Layer(input_dim=self.hidden_size,output_dim=self.output_size,name='output',activation=output_activation)) #output layer


        #hyperparams

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) if optimizer == None else optimizer
        self.print_every=print_every
        self.losses_dict={'categorical_crossentropy':tf.losses.categorical_crossentropy,
                          'mse':tf.losses.mse,'mae':tf.losses.mae,'l1+l2': lambda x,y: tf.losses.mse(x,y)+tf.losses.mae(x,y)}
        self.loss=self.losses_dict[loss]


    def forward_pass(self,data,training=None):
        x = data
        for layer in self.layers:
            x=layer(x,training)
        return x

    def predict(self,X_test):
      return self.forward_pass(X_test)

    def evaluate(self,X_val,y_val,already_one_hot=False):
      preds = self.predict(X_val)

      if self.to_one_hot:
        if not already_one_hot:
          y_val = tf.one_hot(y_val,10)
        num_correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y_val, 1))
      else:
        y_val=tf.Variable(y_val)
        preds = tf.cast(tf.reshape(tf.round(preds),y_val.shape),tf.uint8)
        num_correct_preds = tf.equal(preds,y_val)
      accuracy = tf.reduce_mean(tf.cast(num_correct_preds, tf.float32))

      return accuracy

    def fit(self,x_train,y_train,epochs=5):
      #split into train and validation
        X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=random_seed)

        #convert to tf variables
        X_train = tf.Variable(X_train,dtype=tf.float32)
        X_val = tf.Variable(X_val,dtype=tf.float32)
        if self.to_one_hot:
          Y_train = tf.one_hot(Y_train,10)
          Y_val = tf.one_hot(Y_val,10)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(buffer_size=1000).batch(self.batch_size)
        old_val_acc=0
        val_accs=[]
        for i in range(epochs):
            print(f"epoch :{(i+1):2d}/{epochs} :",end="")
            for batch in train_dataset:
              data,labels = batch
              with tf.GradientTape() as tape:
                 y_pred = self.forward_pass(data,training=True)
                 labels = tf.cast(labels,y_pred.dtype)
                 loss = tf.reduce_mean(self.loss( labels,y_pred))

              grads = tape.gradient(loss, self.trainable_variables)
              self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            train_accuracy = self.evaluate(X_train,Y_train,already_one_hot=self.to_one_hot)
            val_accuracy = self.evaluate(X_val,Y_val,already_one_hot=self.to_one_hot)
            val_accs.append(val_accuracy)
            print(f"Loss = {loss:.3f} Train accuracy={train_accuracy:.3f} Validation accuracy ={val_accuracy:.3f}")
            if tf.abs(old_val_acc-val_accuracy) <=1e-5:

              print("Stopping training as val acc did not improve.")
              break
            old_val_acc=val_accuracy
        return val_accs

"""# Preparing Data and Ground Truth models

## Data Preparation
"""
def gen_train_and_test(i):
  perm = task_permutation[i]
  Trains = x_train[:,perm]
  Tests = x_test[:,perm]

  return Trains,Tests

epochs_to_train=[50]+(num_tasks_to_run-1)*[20]

"""## Ground Truth Models

To Calculate TBWT we need the test performance of num_tasks ground truth models
"""

print("Generating ground truth models on each task")
G=np.zeros(num_tasks_to_run)

for i in range(num_tasks_to_run):
  print(f"\nTraining model for task {i} :\n")
  X_train,X_test = gen_train_and_test(i)

  model = MLP(num_layers=3,input_size=784,output_size=10,loss='categorical_crossentropy')
  model.fit(x_train=x_train,y_train=y_train,epochs=50)
  X_test=tf.Variable(x_test.reshape(-1,784),dtype=tf.float32)
  G[i]=model.evaluate(X_test,y_test)

print(f" The independent classifier performances are : {G}")

"""# The Experiment"""

def run_experiment(model):
  Rt=np.zeros((num_tasks_to_run,num_tasks_to_run))
  metrics={}
  for i in range(num_tasks_to_run):
    print(f'\nTraining on task {i}:')
    X_train,_=gen_train_and_test(i)
    model.fit(x_train=x_train,y_train=y_train,epochs=epochs_to_train[i])


    test_accs=np.zeros(num_tasks_to_run)
    for j in range(num_tasks_to_run):
    #test
      _,X_test = gen_train_and_test(j)
      X_test=tf.Variable(X_test.reshape(-1,784),dtype=tf.float32)
      test_accs[j]=model.evaluate(X_test,y_test)

    Rt[i]=test_accs.copy()

  metrics['Rt']=Rt
  metrics['ACC'] = np.mean(Rt[-1])
  metrics['BWT'] = np.sum(Rt[-1,:] -Rt[np.diag_indices(num_tasks_to_run)])/(num_tasks_to_run-1)
  CBWT = np.zeros(num_tasks_to_run)
  for i in range(num_tasks_to_run-1):
    CBWT[i]=np.sum(Rt[i+1:,i]-Rt[i,i])/(num_tasks_to_run-1-i)
  metrics['CBWT'] = CBWT
  metrics['TBWT'] = np.sum(Rt[-1,:num_tasks_to_run-1]-G[:num_tasks_to_run-1])/(num_tasks_to_run-1)

  return metrics

"""## Effect of Depth on Forgetting"""

models = [MLP(num_layers=2,input_size=784,output_size=10,loss='categorical_crossentropy'),
          MLP(num_layers=3,input_size=784,output_size=10,loss='categorical_crossentropy'),
          MLP(num_layers=4,input_size=784,output_size=10,loss='categorical_crossentropy')]

metrics=np.zeros_like(models)
print("Effect of depth on forgetting")
for i,model in enumerate(models):
  print(f"Training model :{i} with depth :{model.num_layers}")
  metrics[i] =run_experiment(model)

fig = plt.figure(figsize = (20,20)) # width x height
ax=[]

for i in range(len(metrics)):
  Rt=metrics[i]['Rt']
  Rt=np.around(Rt, decimals=2)
  ax.append(fig.add_subplot(3, 3, i+1))
  seaborn.heatmap(Rt,ax=ax[i],annot=True)

plt.show()

for i in range(len(metrics)):
  print(f"Model {i} metrics :")
  print(f"{'Depth':10}:{i+2:>7}")
  print(f"{'ACC':10}:{metrics[i]['ACC']:.3f}")
  print(f"{'BWT':10}:{metrics[i]['BWT']:.3f}")
  print(f"{'TBWT':10}:{metrics[i]['TBWT']:.3f}")
  CBWT = np.around(metrics[i]['CBWT'], decimals=2)
  print(f"{'CBWT':10}:{CBWT}")
  print(f"{'CBWT [t=1]':10}:{CBWT[0]}")

"""The depth 2 model has the best ACC. Even when comparing BWT, TWBT, CBWT, the depth 2 model has best metrics.

 Deeper models are more forgetful

## Effect of Loss Function
"""

loss_fns=['categorical_crossentropy','mse','mae','l1+l2']
output_sizes = [10,1,1,1]
models = [MLP(num_layers=3,input_size=784,output_size=o,loss=l) for o,l in zip(output_sizes,loss_fns)]

metrics=np.zeros_like(models)
print("Effect of loss function on forgetting")

for i,model in enumerate(models):
  print(f"Training model :{i} wiht loss :{loss_fns[i]}")
  metrics[i] =run_experiment(model)

fig = plt.figure(figsize = (15,15)) # width x height
ax=[]

for i in range(len(metrics)):
  Rt=metrics[i]['Rt']
  Rt=np.around(Rt, decimals=2)
  ax.append(fig.add_subplot(4, 4, i+1))
  seaborn.heatmap(Rt,ax=ax[i],annot=True)

plt.show()

for i in range(len(metrics)):
  print(f"Model {i} metrics :")
  print(f"{'Loss':10}:{loss_fns[i]}")
  print(f"{'ACC':10}:{metrics[i]['ACC']:.3f}")
  print(f"{'BWT':10}:{metrics[i]['BWT']:.3f}")
  print(f"{'TBWT':10}:{metrics[i]['TBWT']:.3f}")
  CBWT = np.around(metrics[i]['CBWT'], decimals=2)
  print(f"{'CBWT':10}:{CBWT}")
  print(f"{'CBWT [t=1]':10}:{CBWT[0]}")

"""## Effect of Dropout"""

dropout_rates=[0,0.33,0.5]
models = [MLP(num_layers=3,input_size=784,output_size=10,dropout_rate=rate) for rate in dropout_rates]

metrics=np.zeros_like(models)
print("Effect of dropout on forgetting")
for i,model in enumerate(models):
  print(f"Training model :{i} with dropout :{model.dropout_rate}")
  metrics[i] =run_experiment(model)

fig = plt.figure(figsize = (20,20)) # width x height
ax=[]

for i in range(len(metrics)):
  Rt=metrics[i]['Rt']
  Rt=np.around(Rt, decimals=2)
  ax.append(fig.add_subplot(3, 3, i+1))
  seaborn.heatmap(Rt,ax=ax[i],annot=True)

plt.show()

for i in range(len(metrics)):
  print(f"Model {i} metrics :")
  print(f"{'Dropout Rate':10}:{dropout_rates[i]}")
  print(f"{'ACC':10}:{metrics[i]['ACC']:.3f}")
  print(f"{'BWT':10}:{metrics[i]['BWT']:.3f}")
  print(f"{'TBWT':10}:{metrics[i]['TBWT']:.3f}")
  CBWT = np.around(metrics[i]['CBWT'], decimals=2)
  print(f"{'CBWT':10}:{CBWT}")
  print(f"{'CBWT [t=1]':10}:{CBWT[0]}")

"""## Effect of Optimizer"""

optimizers={'Adam':tf.keras.optimizers.Adam(learning_rate=1e-3),
            'SGD':tf.keras.optimizers.SGD(learning_rate=1e-3,nesterov=True,momentum=0.5),
            'RMSprop':tf.keras.optimizers.RMSprop(learning_rate=1e-3,momentum=0.5)
            }
optimizers_list=list(optimizers.items())
models = [MLP(num_layers=3,input_size=784,output_size=10,optimizer=optimizers[k]) for k,v in optimizers_list]

metrics=np.zeros_like(models)
print("Effect of optimizer on forgetting")

for i,model in enumerate(models):
  print(f"Training model :{i} with optimizer :{optimizers_list[i]}")
  metrics[i] =run_experiment(model)

fig = plt.figure(figsize = (20,20)) # width x height
ax=[]

for i in range(len(metrics)):
  Rt=metrics[i]['Rt']
  Rt=np.around(Rt, decimals=2)
  ax.append(fig.add_subplot(3, 3, i+1))
  seaborn.heatmap(Rt,ax=ax[i],annot=True)

plt.show()

for i in range(len(metrics)):
  print(f"Model {i} metrics :")
  print(f"{'Optimizer':10}:{optimizers_list[i][0]}")
  print(f"{'ACC':10}:{metrics[i]['ACC']:.3f}")
  print(f"{'BWT':10}:{metrics[i]['BWT']:.3f}")
  print(f"{'TBWT':10}:{metrics[i]['TBWT']:.3f}")
  CBWT = np.around(metrics[i]['CBWT'], decimals=2)
  print(f"{'CBWT':10}:{CBWT}")
  print(f"{'CBWT [t=1]':10}:{CBWT[0]}")

"""## Val accuracy plots"""

model = MLP(num_layers=2,input_size=784,output_size=10,loss='categorical_crossentropy')
print("obtaining val accs for each task")
task_vals=[]
for i in range(num_tasks_to_run):
  print(f'\nTraining on task {i}:')
  X_train,_ =gen_train_and_test(i)
  task_val=model.fit(x_train=X_train,y_train=y_train,epochs=epochs_to_train[i])
  task_vals.append(task_val)

<<<<<<< HEAD
plt.figure(figsize=(20,10))
for i in range(num_tasks_to_run):
  val_accs=task_vals[i]
  plt.plot(np.arange(len(val_accs)),val_accs,'-o',label=f'Task {i}')
plt.legend()
plt.show()
=======
#Based on tutorial provided create your MLP model for above problem
#For TF2.0 users Keras can be used for loading trainable variables and dataset.
#You might need google collab to run large scale experiments
>>>>>>> master
