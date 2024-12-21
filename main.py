import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from operator import itemgetter
from keras.src.metrics.accuracy_metrics import accuracy
from keras.src.trainers.trainer import model_supports_jit

mnist = tf.keras.datasets.mnist

training = True



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

model_path = 'keras_flat'
model_name = model_path + '/simple_flat.keras'
if training:
    model = tf.keras.models.Sequetial()
    model.add(tf.keras.layers.Flatten(input_shape = (28,28)))

    model.compile()



    # model = tf.keras.models.Sequential()
    # # First we flatten, add two hidden, and end.
    # model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
    #
    # model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    # model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    # model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
    #
    # model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    #
    # model.fit(x_train, y_train, epochs = 7)
    #
    # model.save(model_name)
    # l,a = model.evaluate(x_test, y_test)
    # print(f'Accuracy:{a}\nLoss:{l}')
    # print(model.summary())


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

def evaluate_model(m):
    model = tf.keras.models.load_model(m)
    loss, accuracy = model.evaluate(x_test, y_test)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    return accuracy,loss

getEfficiency = lambda l: (l[0]/l[1])*10**8

def display_rank(l, metric, metric_index, invert = False):
    print(f'\nRanked by {metric}')
    if invert:
        l.reverse()
    for i in range(len(l)):
        output = f'{i+1}:{l[i][0][:-6]} {metric}: [{l[i][metric_index]}]'
        if metric != 'Efficiency':
            output += f'\t\t Efficiency:{str(getEfficiency([l[i][1], l[i][3]]))}'
        print(output)


def evaluate_all_models(directory):
    model_rank = []
    a,l = 0,0
    for filename in os.listdir(directory):

        model_name = str(filename)
        model_details = model_name[:-7]
        epoch_number = model_details[len(model_details)-1]
        hidden_layer_number = model_details[0]
        layer_details = model_details[2:-2]
        multiplication_count = 1
        if '&' in layer_details:
            vals = layer_details.split('&')
            new_details = ''

            for i in range(len(vals)):
                multiplication_count *= int(vals[i])
                new_details += '\n\th'+str(i+1)+': '+vals[i]
            layer_details = new_details
        else:
            for i in range(int(hidden_layer_number)):
                if 'dropout' not in layer_details:
                    multiplication_count *= int(layer_details)
                else:
                    multiplication_count *= 128
        model_description = f'Hidden Count:{str(hidden_layer_number)}\tEpoch Count:{epoch_number}\tLayer Size:{layer_details}'
        print()
        print(model_description)
        print(f'Multiplications between hidden layers: {multiplication_count}')
        a,l=evaluate_model(model_path+'/'+model_name)
        flop = 28*28*multiplication_count*10
        print(f'FLOP: {flop}')
        model_rank.append([model_name, a, l, flop, getEfficiency([a, flop])])

    ranked_by_accuracy = sorted(model_rank, key = itemgetter(1))
    ranked_by_loss = sorted(model_rank, key = itemgetter(2))
    ranked_by_efficiency = sorted(model_rank, key = itemgetter(4))
    display_rank(ranked_by_accuracy, 'Accuracy', 1, True)
    display_rank(ranked_by_loss, 'Loss', 2)
    display_rank(ranked_by_efficiency, 'Efficiency', 4, True)

    # for i in range(len(ranked_by_accuracy)):
    #     print(f'{i}:{ranked_by_accuracy[i][0]} [accuracy:{ranked_by_accuracy[i][1]}] with accuracy/flop:{(a)/(ranked_by_accuracy[i][3])*(10**8)}')


def evaluate_nonflat():
    a, l = 0, 0
    for filename in os.listdir('keras_nonflat'):
        model_name = str(filename)
        evaluate_model('keras_nonflat'+'/'+filename)


evaluate_all_models('keras_flat') if not training else 0
