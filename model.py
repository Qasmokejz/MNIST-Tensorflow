import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 

filename = 'mod1'

# train_X.shape = (60000, 28, 28)
# train_y.shape = (60000,)
(train_x, train_y), (test_x, test_y) = mnist.load_data()

#flatten train_X to train_x (60000, 784)
#train_x = np.array([i.flatten() for i in train_X])

#model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['Accuracy']
)

r = model.fit(
    train_x,
    train_y,
    epochs = 15
)

model.save(filename)

#plot 

plt.subplot(1,2,1)
plt.plot(r.history['loss'])

plt.subplot(1,2,2)
plt.plot(r.history['Accuracy'])

plt.show()
