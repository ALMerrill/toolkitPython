from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix

import numpy as np

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
        NUM_OUTPUT_CLASSES = 3
        SCALE = 5
        MAX_EPOCHS = 1
        STOP_EPOCHS = 5
        ERROR_BOUND = .02
        self.LAYERS = 3
        self.c = .1
        self.bias = [1]
        self.input_size = 3
        self.output_size = 2
        self.hiddenSize = 2
        # self.bias_list = [
        #     np.random.randn(1,features.cols*2) / SCALE,
        #     np.random.randn(1,NUM_OUTPUT_CLASSES) / SCALE ]
        self.weight_list = [
            np.random.randn(features.cols+1, features.cols*2) / SCALE,
            np.random.randn((features.cols*2)+1, NUM_OUTPUT_CLASSES) / SCALE ]

        self.initialize_to_ex()
        # self.act_list = [
        #     np.ones((1,features.cols*2)),
        #     np.ones((1,NUM_OUTPUT_CLASSES)) ]
        self.act_list = [np.ones(2),np.ones(2),np.ones(2)]
        self.error_list = [np.ones(2),np.ones(2),np.ones(2)]
        # self.delta_w_list = [
        #     np.ones((1,features.cols*2)),
        #     np.ones((1,NUM_OUTPUT_CLASSES)) ]
        # self.error_list = [
        #     np.ones((1,features.cols*2)),
        #     np.ones((1,NUM_OUTPUT_CLASSES)) ]

        for epochs in range(MAX_EPOCHS):
            # training_set, training_labels, validation_set, validation_labels = self.split_data(features, labels)
            self.one_training_epoch(features.data, labels.data)

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        labels += self.labels

    def one_training_epoch(self, features, labels):
        for n in range(len(features)):
            self.inputs = features[n][:len(features[n])//2+1] + self.bias
            self.targets = features[n][len(features[n])//2+1:] + labels[n]
            self.forward(features, labels, n)
            self.error()

    def forward(self, features, labels, n):
        self.print_weights(self.weight_list)
        inputs = features[n][:len(features[n])//2+1] + self.bias
        print(inputs)
        for l in range(self.LAYERS):
            net = self.weight_list[l].dot(inputs)
            self.act_list[l] = self.act(net)
            # inputs = np.insert(self.act_list[l], 0, self.bias)
            inputs = np.append(self.act_list[l], self.bias)

        print("features",features[n][:len(features[n])//2+1])
        print("Inputs:", inputs)
        print("targets:", self.targets)
        print("predicted output: ",end="")
        for act in self.act_list[self.LAYERS - 1]:
            print(str.format('{0:.10f}', act), end=", ")
        print("\n")

    def error(self):
        #last layer of error list
        outputs = self.act_list[self.LAYERS-1]
        diff = self.targets - outputs
        self.error_list[self.LAYERS-1] = diff * self.actPrime(outputs)
        #last layer of weight change list
        # act_buf = np.insert(self.act_list[self.LAYERS-1], 0, self.bias)
       
        for l in reversed(range(self.LAYERS-1)):
            #calculate error
            # tot = []
            # for i in range(2):
            #     tot.append(self.weight_list[l+1][:,i].dot(self.error_list[l+1]))
            prod = self.error_list[l+1].reshape(1,2).dot(self.weight_list[l+1][:,:self.LAYERS-1])
            prime_l = self.actPrime(self.act_list[l])
            self.error_list[l] = np.multiply(prod,prime_l).reshape(2,)

            #calculate weight change
            # act_buf = np.append(self.act_list[l-1], self.bias)
            # self.delta_w_list[l] = self.c * (self.error_list[l].reshape(2,1) * act_buf)

        #calculate weight change dependent on node output
        for l in reversed(range(self.LAYERS)):
            if l > 0: #weights dependent on node output
                act_buf = np.append(self.act_list[l-1], self.bias)
                self.delta_w_list[l] = self.c * (self.error_list[l].reshape(2,1) * act_buf)
            else:   #weights dependent on inputs
                self.delta_w_list[l] = self.c * (self.error_list[l].reshape(2,1) * self.inputs)

        for l in range(len(self.weight_list)):
            self.weight_list[l] = self.weight_list[l] + self.delta_w_list[l]
        self.print_error(self.error_list)
        self.print_weights(self.weight_list)

    def backward(self):
        pass
        
    def initialize_to_ex(self):
        # weight_list each row is a node with it's incoming weights
        # add a bias weight to each row
        # bias_list is just a bunch of ones
        self.weight_list = [
            # np.array([[.1,.2,-.1],[-.2,.3,-.3]]),
            # np.array([[.1,-.2,-.3],[.2,-.1,.3]]),
            # np.array([[.2,-.1,.3],[.1,-.2,-.3]]) ]
            np.array([[.2,-.1,.1],[.3,-.3,-.2]]),
            np.array([[-.2,-.3,.1],[-.1,.3,.2]]),
            np.array([[-.1,.3,.2],[-.2,-.3,.1]]) ]

        self.delta_w_list = [
            np.array([[0,0,0],[0,0,0]]),
            np.array([[0,0,0],[0,0,0]]),
            np.array([[0,0,0],[0,0,0]]) ]

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
                print(str.format('{0:.13f}', error[i][j]), end=", ")
            print()

    def split_data(self, features, labels):
        features.shuffle(labels)
        num_rows = features.rows
        training_set = features.data[:int(num_rows*.8)]
        training_labels = labels.data[:int(num_rows*.8)]
        validation_set = features.data[int(num_rows*.8):]
        validation_labels = labels.data[int(num_rows*.8):]
        return training_set, training_labels, validation_set, validation_labels

    def act(self, net):
        return 1 / (1 + np.exp(-net))

    def actPrime(self, out):
        return out * (1 - out)

    def evaluate_error(self, features, labels):
        pass


