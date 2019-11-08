import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

#Load and preprocess MNIST dataset
(train_x, train_y), (test_x,test_y) = tf.keras.datasets.mnist.load_data()
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)


#------------------------Build-model-----------------------------
model = tf.keras.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))
"""

Add your layers here!!

"""
model.add(layers.Dense(10,activation='sigmoid'))
#------------------------Build-model-----------------------------


#Set learning rate, optimizer, loss, epochs and batch_size
opti = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opti, loss='mse', metrics=['accuracy'])
history = model.fit(train_x, train_y, batch_size=20,epochs=20,validation_split=0.16)

#plot the training result
record = history.history
epochs = history.epoch
plt.figure(figsize=(10,10))
plt.xlabel('epoch')
plt.ylabel('losses')
plt.plot(epochs, record['loss'],'b', label='loss')
plt.plot(epochs, record['val_loss'],'r', label='val_loss')
plt.legend()
plt.show()

#evaluate the model with new data
model.evaluate(test_x,test_y)
