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
        MAX_EPOCHS = 3
        # MAX_EPOCHS = 1
        STOP_EPOCHS = 5
        ERROR_BOUND = .02
        self.LAYERS = 2
        # self.LAYERS = 3
        self.c = .175
        # self.c = .1
        self.m = .9
        # self.m = 0
        self.bias = [1]
        self.layerSizes = [2,3,1]
        self.errorLayers = [3,1] 
        # self.layerSizes = [2, 2, 2, 2]
        # self.errorLayers = [2, 2, 2]
        # self.bias_list = [
        #     np.random.randn(1,features.cols*2) / SCALE,
        #     np.random.randn(1,NUM_OUTPUT_CLASSES) / SCALE ]
        # self.weight_list = [
        #     np.random.randn(features.cols+1, features.cols*2) / SCALE,
        #     np.random.randn((features.cols*2)+1, NUM_OUTPUT_CLASSES) / SCALE ]
        # self.act_list = [
        #     np.ones((1,features.cols*2)),
        #     np.ones((1,NUM_OUTPUT_CLASSES)) ]
        # self.delta_w_list = [
        #     np.ones((1,features.cols*2)),
        #     np.ones((1,NUM_OUTPUT_CLASSES)) ]
        # self.error_list = [
        #     np.ones((1,features.cols*2)),
        #     np.ones((1,NUM_OUTPUT_CLASSES)) ]
        self.initialize_to_ex()
        
        self.print_weights(self.weight_list)

        for epochs in range(MAX_EPOCHS):
            print("---Epoch " + str(epochs + 1) + "---")
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
        # for n in range(2):
            self.inputs = features[n][:len(features[n])//2+1] + self.bias
            self.targets = features[n][len(features[n])//2+1:] + labels[n]
            self.forward(features, labels, n)
            self.backward()

    def forward(self, features, labels, n):
        inputs = features[n][:len(features[n])//2+1] + self.bias
        print("Pattern:", inputs)
        print("Forward propagating...")
        for l in range(self.LAYERS):
            net = self.weight_list[l].dot(inputs)
            # print("inputs",)
            # self.print_list(inputs)
            # print("weights", self.weight_list[l])
            # print("net", net)
            self.act_list[l] = self.act(net)
            inputs = np.append(self.act_list[l], self.bias)

        print("predicted output: ",end="")
        for l in reversed(range(self.LAYERS)):
            for act in self.act_list[l]:
                print(str.format('{0:.14f}', act), end=", ")
        print("\n")

    def backward(self):
        print("Back propagating...")
        #last layer of error list
        outputs = self.act_list[self.LAYERS-1]
        diff = self.targets - outputs
        self.error_list[self.LAYERS-1] = diff * self.actPrime(outputs)
       
        for l in reversed(range(self.LAYERS-1)):
            # print("outputs:",self.act_list[l])
            # print("error_k:", self.error_list[l+1])
            # print("weights:",self.weight_list[l+1])
            # print("weights_jk", self.weight_list[l+1][0][:self.errorLayers[l]])
            print("weights_l+1:",self.weight_list[l+1])
            print("weights_l+1 without bias:",self.weight_list[l+1][:,:-1])
            prod = self.error_list[l+1].dot(self.weight_list[l+1][:,:-1])
            # prod = self.error_list[l+1] * self.weight_list[l+1][0][:self.errorLayers[l]]
            # print("prod:",prod)
            prime_l = self.actPrime(self.act_list[l])
            # print("prime_l",prime_l)
            self.error_list[l] = np.multiply(prod,prime_l).reshape(self.errorLayers[l],)
            # print("error_l:",self.error_list[l])
        self.print_error(self.error_list)        

        print("Descending Gradient...")
        for l in reversed(range(self.LAYERS)):
            reshaped_error = self.error_list[l].reshape(self.errorLayers[l],1)
            if l > 0: #weights dependent on node output
                act_buf = np.append(self.act_list[l-1], self.bias)
                self.delta_w_list[l] = self.c * (reshaped_error * act_buf) + self.m * self.delta_w_list[l]
            else:   #weights dependent on inputs
                self.delta_w_list[l] = self.c * (reshaped_error * self.inputs) + self.m * self.delta_w_list[l]

        for l in range(len(self.weight_list)):
            self.weight_list[l] = self.weight_list[l] + self.delta_w_list[l]
        self.print_weights(self.weight_list)
        
    def initialize_to_ex(self):
        self.weight_list = [
            np.array([[-.03,.03,-.01],[.04,-.02,.01],[.03,.02,-.02]]),
            np.array([[-.01,.03,.02,.02]]) ]

        self.delta_w_list = [
            np.array([[0,0,0],[0,0,0],[0,0,0]]),
            np.array([[0,0,0,0]]) ]

        self.act_list = [np.ones(2),np.ones(2),np.ones(2)]
        self.error_list = [np.ones(3),np.ones(1)]
        # self.act_list = [np.ones(2),np.ones(2),np.ones(2)]
        # self.error_list = [np.ones(2),np.ones(2),np.ones(2)]
        # self.weight_list = [
        #     np.array([[.2,-.1,.1],[.3,-.3,-.2]]),
        #     np.array([[-.2,-.3,.1],[-.1,.3,.2]]),
        #     np.array([[-.1,.3,.2],[-.2,-.3,.1]]) ]

        # self.delta_w_list = [
        #     np.array([[0,0,0],[0,0,0]]),
        #     np.array([[0,0,0],[0,0,0]]),
        #     np.array([[0,0,0],[0,0,0]]) ]


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

    def act(self, net):
        return 1 / (1 + np.exp(-net))

    def actPrime(self, out):
        return out * (1 - out)

    def evaluate_error(self, features, labels):
        pass


