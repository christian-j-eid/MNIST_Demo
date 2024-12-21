import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from autograd import grad
import autograd.numpy as npg
from time import perf_counter


class NeuralNetwork:
    #Creating Neural Network Based of
    def __init__(self):
        self.x_train = self.normalizePixels(x_train)
        self.y_train = y_train
        self.x_test = self.normalizePixels(x_test)
        self.y_test = y_test
        self.OUTPUT_SIZE = 10
        self.net = []
        # print(self.x_train[0].size)
        # print(self.INPUT_SIZE)

        self.generateNetwork()
        self.printNetworkShapes()


    def generateNetwork(self):

        w_input_h1 = self.generateWeights(784, 64)
        w_h1_h2 = self.generateWeights(64, 64)
        w_h2_out = self.generateWeights(64, 9)
        layer_1 =  self.generateLayer(64)
        layer_2 = self.generateLayer(64)
        output_layer = self.generateLayer(9)
        self.net = [w_input_h1, layer_1, w_h1_h2, layer_2, w_h2_out, output_layer]

    # Building
    # Input comes in as 28*28 grid
    # flattened ot a (!,784) input,

    # for all non input layer
    # input[i] = ∂(x[i]*w[i]+b[i])
    # where ∂ is relu for hidden layers, sparse_categoricalCrossEntropy for output
    # we can set arbitrary (1st or last) x to be 1, and add an extra weight which acts as bias

    #INPUT TO LAYER 1
    #(1,784)

    def printNetworkShapes(self):
        titles = ['W1', 'H1', 'W2', 'H2','W3','Output']
        for i in range(len(titles)):
            print(titles[i],': ', np.array(self.net[i]).shape)

    def generateLayer(self, n):
        output = []
        for i in range(n+1):
            output.append(1.0)
        return output

    def generateWeights(self, m, n):
        output = []
        for i in range(m):
            weights = []
            for j in range(n+1):
                weights.append(np.random.uniform(-0.3, 0.3))
            output.append(weights)
        return output

    def flatten(self,x):
        flattened = []
        for i in range(len(x)):

            image = []
            for row in self.x_train[i]:
                for value in row:
                    image.append(value)
            flattened.append(image)

        return np.array(flattened)

    def normalizePixels(self, inputs):
        return  tf.keras.utils.normalize(inputs, axis=1)


    def ReLu(self, x):
        return max(0, x)

    def ReLu_dx(self, x):
        if x<=0:
            return 0
        else:
            return 1

    def delta_output(self,y,a):
        return -(y - a)

    def forwardPropogate(self):
        w1, h1, w2, h2, w3, output = self.net

    def backward_pass(self):
        pass

    def evaluate_model(self):
        pass


Model1 = NeuralNetwork()
