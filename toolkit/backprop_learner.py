from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix

import matplotlib.pyplot as plt
import numpy as np
import sys
from copy import deepcopy

class BackpropLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    labels = []

    def __init__(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.data = "other"
        if sys.argv[4] == 'datasets/iris.arff':
            self.data = 'iris'
        elif sys.argv[4] == 'datasets/vowel.arff':
            self.data = 'vowel'


        #hyperparameters
        self.c = 0.00075
        self.m = .9
        self.bias = [1]
        self.act_func = 'tanh' #options: ['sigmoid', 'tanh']
        SCALE = 100
        MAX_EPOCHS = 3000
        STOP_EPOCHS = 20
        ERROR_BOUND = .03
        graph_error = False

        #keep track of the weights that produced the least error
        least_weights = []
        epochs = 1
        total_epochs = 0
        err = 1
        least_error = 1
        best_epoch = 0

        #graph lists
        if graph_error:
            t_mse = []
            v_mse = []
            class_rate = []


        if self.data == 'iris':
            self.featuresUsed = np.array([0,1,2,3])
            self.layerSizes = [len(self.featuresUsed),8,3]

        elif self.data == 'vowel':
            self.featuresUsed = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
            self.layerSizes = [len(self.featuresUsed),32,11]

        elif self.data == 'cancer':
            self.featuresUsed = np.array([0,1,2,3,4,5,6,7,8])

        NUM_OUTPUT_CLASSES = self.layerSizes[-1]
        self.errorLayers = self.layerSizes[1:]
        self.LAYERS = len(self.errorLayers)

        #neural net matrices
        self.weight_list = [
            np.random.randn(self.layerSizes[1], self.layerSizes[0]+1) / SCALE,
            np.random.randn(NUM_OUTPUT_CLASSES, (self.layerSizes[1])+1) / SCALE ]
        self.act_list = [
            np.ones((1,self.layerSizes[1])),
            np.ones((1,NUM_OUTPUT_CLASSES)) ]
        self.delta_w_list = [
            np.zeros(self.weight_list[0].shape),
            np.zeros(self.weight_list[1].shape) ]
        self.error_list = [
            np.ones((1,self.layerSizes[1])),
            np.ones((1,NUM_OUTPUT_CLASSES)) ]

        for epoch in range(MAX_EPOCHS):
            total_epochs += 1

            training_set, training_labels = self.get_training_set(features, labels)
            t_set, t_labels, v_set, v_labels = self.split_data_predict(features, labels)
            self.one_training_epoch(training_set, training_labels)
            error_rate = (1 - self.measure_accuracy(v_set, v_labels))
            if graph_error:
                t_mse.append(self.mse(t_set, t_labels))
                v_mse.append(self.mse(v_set, v_labels))
                class_rate.append(1 - error_rate)

            #evaluate stopping criteria
            if error_rate < least_error:
                least_error = error_rate
                least_weights = deepcopy(self.weight_list)
                best_epoch = epoch
                print("training:",1 - self.measure_accuracy(t_set,t_labels))
                print("validation:",error_rate)
            if abs(error_rate - err) < ERROR_BOUND:
                epochs += 1
                if epochs == STOP_EPOCHS:
                    if error_rate > least_error:
                        error_rate = least_error
                        self.weight_list = least_weights
                        if graph_error:
                            class_rate[-1] = 1-error_rate
                            t_mse[-1] = min(t_mse)
                            v_mse[-1] = min(v_mse)

                    break
            else:
                epochs = 0
                err = error_rate
        print("TOTAL EPOCHS:",total_epochs)
        print("BEST EPOCH:", best_epoch)

        if graph_error:
            fig, ax1 = plt.subplots()
            x = np.arange(0.0, total_epochs, 1.0)
            ts, = ax1.plot(x, t_mse, 'b-', label="Training")
            vs, = ax1.plot(x, v_mse, 'g-', label="Validation")
            ax1.set_xlabel('Epochs')
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylabel('MSE', color='b')
            ax1.tick_params('y', colors='b')

            ax2 = ax1.twinx()
            cr, = ax2.plot(x, class_rate, 'r-')
            ax2.set_ylabel('Classification rate', color='r')
            ax2.tick_params('y', colors='r')

            fig.tight_layout()
            plt.legend(handles=[ts,vs,cr])
            plt.show()

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        self.forward(features, labels, -1)
        labels += [np.argmax(self.act_list[self.LAYERS-1])]

    def one_training_epoch(self, features, labels):
        features = np.array(features)
        for n in range(len(features)):
            self.inputs = np.append(features[n][self.featuresUsed], self.bias)
            self.targets = [0 for _ in range(self.layerSizes[-1])]
            self.targets[int(labels[n][0])] = 1
            self.forward(features, labels, n)
            self.backward()

    def forward(self, features, labels, n = -1):
        if n == -1:
            features = np.array(features)
        else:
            features = np.array(features[n])
        inputs = np.append(features[self.featuresUsed], self.bias)
        for l in range(self.LAYERS):
            net = np.matmul(self.weight_list[l], inputs)
            self.act_list[l] = self.act(net)
            inputs = np.append(self.act_list[l], self.bias)

    def backward(self):
        #output layer
        outputs = self.act_list[-1]
        diff = self.targets - outputs
        self.error_list[self.LAYERS-1] = diff * self.actPrime(outputs)
       
        #hidden layer(s)
        for l in reversed(range(self.LAYERS-1)):
            prod = self.error_list[l+1].dot(self.weight_list[l+1][:,:-1])
            prime_l = self.actPrime(self.act_list[l])
            self.error_list[l] = np.multiply(prod,prime_l).reshape(self.errorLayers[l],)

        for l in reversed(range(self.LAYERS)):
            reshaped_error = self.error_list[l].reshape(self.errorLayers[l],1)
            if l > 0: #weights dependent on node output
                act_buf = np.append(self.act_list[l-1], self.bias)
                prod = reshaped_error * act_buf
                self.delta_w_list[l] = self.c * (reshaped_error * act_buf) + self.m * self.delta_w_list[l]
            else:   #weights dependent on inputs
                self.delta_w_list[l] = self.c * (reshaped_error * self.inputs) + self.m * self.delta_w_list[l]

        for l in range(len(self.weight_list)):
            self.weight_list[l] = self.weight_list[l] + self.delta_w_list[l]

    def mse(self, features, labels):
        pred = [0]
        sse = 0.0
        for i in range(features.rows):
            feat = features.row(i)
            targ = labels.row(i)
            pred[0] = 0.0
            self.predict(feat, pred)
            delta = targ[0] - pred[0]
            sse += delta**2
        return sse / features.rows

    def get_training_set(self, features, labels):
        features.shuffle(labels)
        num_rows = features.rows

        training_set = features.data[:int(num_rows*.8)]
        training_labels = labels.data[:int(num_rows*.8)]
        return training_set, training_labels

    def split_data_predict(self, features, labels):
        features.shuffle(labels)
        num_rows = features.rows
        training_set = deepcopy(features)
        training_labels = deepcopy(labels)
        validation_set = deepcopy(features)
        validation_labels = deepcopy(labels)
        training_set.data = features.data[:int(num_rows*.8)]
        training_labels.data = labels.data[:int(num_rows*.8)]
        validation_set.data = features.data[int(num_rows*.8):]
        validation_labels.data = labels.data[int(num_rows*.8):]
        return training_set, training_labels, validation_set, validation_labels

    def act(self, net):
        if self.act_func == 'sigmoid':
            return 1 / (1 + np.exp(-net))
        elif self.act_func == 'tanh':
            return np.tanh(net)
        elif self.act_func == 'ReLU':
            lessThanZero = net < 0
            net[lessThanZero] = 0
            return net
        else:
            raise ValueError('Invalid activation function received')

    def actPrime(self, out):
        if self.act_func == 'sigmoid':
            return out * (1 - out)
        elif self.act_func == 'tanh':
            return 1 - (np.tanh(out)**2)
        elif self.act_func == 'ReLU':
            greaterThanZero = out > 0
            lessThanOrEqualToZero = out <= 0
            out[greaterThanZero] = 1
            out[lessThanOrEqualToZero] = 0
            return out
        else:
            raise ValueError('Invalid activation function received')


