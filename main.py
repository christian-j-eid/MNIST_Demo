import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.metrics.accuracy_metrics import accuracy

mnist = tf.keras.datasets.mnist

training = False


# Getting training data & testing data from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def display_digit(sample):
    conversion1 = np.array(sample, dtype='float')
    pixels = conversion1.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


# Normalizing pixel values from 0-255 to 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


if training:
    model = tf.keras.models.Sequential()
    # First we flatten, add two hidden, and end.

    model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
    model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

    model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    model.fit(x_train, y_train, epochs = 3)

    model.save('digit_model1.keras')


model = tf.keras.models.load_model('digit_model1.keras')
loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

def prediction_meaning(p):
    s = 'Prediction:'
    c = 0
    for n in p[0]:
        s += str(c) + ": " + str(n)  + '\n'
        c += 1
    print(s)

def predict_with_model(m, x):
    x = np.array(x).reshape(1, 28, 28)
    prediction = m.predict(x)
    print(prediction)
    prediction_meaning(prediction)

predict_with_model(model, x_test[4])