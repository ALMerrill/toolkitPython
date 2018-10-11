from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix

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
        elif sys.argv[4] == 'datasets/back2.arff':
            self.data = 'back2'
        elif sys.argv[4] == 'datasets/vowel.arff':
            self.data = 'vowel'

        self.method = "other"
        if sys.argv[6] == 'training':
            self.method = 'training'
        SCALE = 20
        MAX_EPOCHS = 1000
        STOP_EPOCHS = 5
        ERROR_BOUND = .02
        self.c = .1
        self.m = .9
        self.bias = [1]

        if self.data == 'iris':
            NUM_OUTPUT_CLASSES = 3
            self.LAYERS = 2
            self.layerSizes = [4,8,3]
            self.errorLayers = [8,3] 
            self.weight_list = [
                np.random.randn(self.layerSizes[1], features.cols+1) / SCALE,
                np.random.randn(NUM_OUTPUT_CLASSES, (self.layerSizes[1])+1) / SCALE ]
            self.act_list = [
                np.ones((1,features.cols*2)),
                np.ones((1,NUM_OUTPUT_CLASSES)) ]
            self.delta_w_list = [
                np.zeros(self.weight_list[0].shape),
                np.zeros(self.weight_list[1].shape) ]
            self.error_list = [
                np.ones((1,features.cols*2)),
                np.ones((1,NUM_OUTPUT_CLASSES)) ]

        elif self.data == 'back2':
            NUM_OUTPUT_CLASSES = 3
            self.LAYERS = 2
            self.layerSizes = [2,3,1]
            self.errorLayers = [3,1]
            self.initialize_to_ex()

        elif self.data == 'vowel':
            NUM_OUTPUT_CLASSES = 11
            self.LAYERS = 2
            self.layerSizes = [13,8,11]
            self.errorLayers = [8,11] 
            self.weight_list = [
                np.random.randn(self.layerSizes[1], features.cols+1) / SCALE,
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

        
        # self.print_weights(self.weight_list)

        ERROR_BOUND = .1
        total_epochs = 0
        epochs = 1
        err = 1
        least_error = 1
        least_weights = []

        for epoch in range(MAX_EPOCHS):
            total_epochs += 1
            if self.data == 'back2':
                print("---Epoch " + str(epoch + 1) + "---")
            
            if self.method == 'training':
                self.one_training_epoch(features.data, labels.data)
            else:
                training_set, training_labels, validation_set, validation_labels = self.split_data(features, labels)
                validation_set, validation_labels = self.split_data_predict(features, labels)
                self.one_training_epoch(training_set, training_labels)
                rmse = (1 - self.measure_accuracy(validation_set, validation_labels))
                # print(1-rmse)

            if rmse < least_error:
                least_error = rmse
                least_weights = deepcopy(self.weight_list)
                # print(least_weights)
            if abs(rmse - err) < ERROR_BOUND:
                epochs += 1
                if epochs == STOP_EPOCHS:
                    if rmse > least_error:
                        rmse = least_error
                        self.weight_list = least_weights
                    break
            else:
                epochs = 0
                err = rmse
        print("TOTAL EPOCHS:",total_epochs)
        # print(self.weight_list)

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        self.predict_forward(features, labels)
        labels += [np.argmax(self.act_list[self.LAYERS-1])]
        # print(self.act_list)

    def one_training_epoch(self, features, labels):
        for n in range(len(features)):
        # for n in range(1):
            if self.data == 'iris' or self.data == 'vowel':
                self.inputs = features[n] + self.bias
                self.targets = [0 for _ in range(self.layerSizes[-1])]
                self.targets[int(labels[n][0])] = 1
            elif self.data == 'back2':
                self.inputs = features[n][:len(features[n])//2+1] + self.bias
                self.targets = features[n][len(features[n])//2+1:] + labels[n]
            self.forward(features, labels, n)
            self.backward()

    def forward(self, features, labels, n):
        if self.data == 'iris' or self.data == 'vowel':
            inputs = features[n] + self.bias
        elif self.data == 'back2':
            inputs = features[n][:len(features[n])//2+1] + self.bias
            print("Pattern:", inputs)
            print("Forward propagating...")
        for l in range(self.LAYERS):
            net = np.matmul(self.weight_list[l], inputs)
            self.act_list[l] = self.act(net)
            inputs = np.append(self.act_list[l], self.bias)

        if self.data == 'back2':
            print("predicted output: ",end="")
            for l in reversed(range(self.LAYERS)):
                for act in self.act_list[l]:
                    print(str.format('{0:.14f}', act), end=", ")
            print("\n")

    def predict_forward(self, features, labels):
        if self.data == 'iris' or self.data == 'vowel':
            inputs = features + self.bias
        elif self.data == 'back2':
            inputs = features[:len(features)//2+1] + self.bias
            
        for l in range(self.LAYERS):
            net = np.matmul(self.weight_list[l], inputs)
            self.act_list[l] = self.act(net)
            inputs = np.append(self.act_list[l], self.bias)

    def backward(self):
        if self.data == 'back2':
            print("Back propagating...")
        #output layer
        outputs = self.act_list[-1]
        diff = self.targets - outputs
        self.error_list[self.LAYERS-1] = diff * self.actPrime(outputs)
       
        #hidden layer(s)
        for l in reversed(range(self.LAYERS-1)):
            prod = self.error_list[l+1].dot(self.weight_list[l+1][:,:-1])
            prime_l = self.actPrime(self.act_list[l])
            self.error_list[l] = np.multiply(prod,prime_l).reshape(self.errorLayers[l],)

        if self.data == 'back2':
            self.print_error(self.error_list)        
            print("Descending Gradient...")

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

        if self.data == 'back2':
            self.print_weights(self.weight_list)

    def evaluate_error(self, features, labels):
        pred = [0]
        sse = 0.0
        for i in range(features.rows):
            feat = features.row(i)
            targ = labels.row(i)
            pred[0] = 0.0       # make sure the prediction is not biased by a previous prediction
            self.predict(feat, pred)
            delta = targ[0] - pred[0]
            sse += delta**2
        return sse / features.rows
        
    def initialize_to_ex(self):
        self.weight_list = [
            np.array([[-.03,.03,-.01],[.04,-.02,.01],[.03,.02,-.02]]),
            np.array([[-.01,.03,.02,.02]]) ]

        self.delta_w_list = [
            np.array([[0,0,0],[0,0,0],[0,0,0]]),
            np.array([[0,0,0,0]]) ]

        self.act_list = [np.ones(3),np.ones(1)]
        self.error_list = [np.ones(3),np.ones(1)]

    def print_list(self, l):
        for element in l:
            print(str.format('{0:.17f}', element), end=", ")
        print()

    def print_weights(self, weights):
        print("weights:")
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                for k in range(len(weights[i][j])):
                    print(str.format('{0:.13f}', weights[i][j][k]), end=", ")
                print()
            print()
    
    def print_error(self, error):
        print("error:")
        for i in range(len(error)):
            for j in range(len(error[i])):
                print(str.format('{0:.17f}', error[i][j]), end=", ")
            print()

    def split_data(self, features, labels):
        features.shuffle(labels)
        num_rows = features.rows

        training_set = features.data[:int(num_rows*.8)]
        training_labels = labels.data[:int(num_rows*.8)]
        validation_set = features.data[int(num_rows*.8):]
        validation_labels = labels.data[int(num_rows*.8):]
        return training_set, training_labels, validation_set, validation_labels

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
        return validation_set, validation_labels

    def act(self, net):
        return 1 / (1 + np.exp(-net))
        # return np.tanh(-net)

    def actPrime(self, out):
        return out * (1 - out)
        # return 1 - (np.tanh(out)**2)

    def evaluate_error(self, features, labels):
        pass


