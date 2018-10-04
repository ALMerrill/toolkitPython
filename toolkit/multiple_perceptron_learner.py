from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix

import random
import numpy as np


class MultiplePerceptronLearner(SupervisedLearner):
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

        self.c = .001
        self.bias = [1]
        self.weights_setosa = [0 for _ in range(features.cols + 1)]     #0
        self.weights_versicolor = [0 for _ in range(features.cols + 1)] #1
        self.weights_virginica = [0 for _ in range(features.cols + 1)]  #2
        MAX_EPOCHS = 1000
        STOP_EPOCHS = 10
        ERROR_BOUND = .15
        total_epochs = 0
        epochs = 1
        err = 1
        least_error = 1
        least_weights_setosa = []
        least_weights_versicolor = []
        least_weights_virginica = []
        for epoch in range(MAX_EPOCHS):
            total_epochs += 1
            training_set, training_labels, validation_set, validation_labels = self.split_data(features, labels)

            self.one_training_epoch(training_set, training_labels)
            error_rate = self.evaluate_error(validation_set, validation_labels)

            if error_rate < least_error:
                least_error = error_rate
                least_weights_setosa = self.weights_setosa
                least_weights_versicolor = self.weights_versicolor
                least_weights_virginica = self.weights_virginica
            if abs(error_rate - err) < ERROR_BOUND:
                epochs += 1
                if epochs == STOP_EPOCHS:
                    if error_rate > least_error:
                        error_rate = least_error
                        self.weights_setosa = least_weights_setosa
                        self.weights_versicolor = least_weights_versicolor
                        self.weights_virginica = least_weights_virginica
                    break
            else:
                epochs = 0
                err = error_rate


    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        inputs = features + self.bias
        output = self.calculate_output(inputs)
        prediction = [output]
        labels += prediction

    def split_data(self, features, labels):
        features.shuffle(labels)
        num_rows = features.rows
        training_set = features.data[:int(num_rows*.8)]
        training_labels = labels.data[:int(num_rows*.8)]
        validation_set = features.data[int(num_rows*.8):]
        validation_labels = labels.data[int(num_rows*.8):]
        return training_set, training_labels, validation_set, validation_labels

    def one_training_epoch(self, features, labels):
        errors = 0  #number of errors in epoch
        for itr in range(len(features)):
            target = int(labels[itr][0])
            targets = [0,0,0]
            targets[target] = 1
            inputs = features[itr] + self.bias
            total_output, outputs = self.calculate_output(inputs, True)
            if total_output != target:
                errors += 1

            deltaW_setosa = self.calculateDeltaW(targets[0], outputs[0], inputs)
            deltaW_versicolor = self.calculateDeltaW(targets[1], outputs[1], inputs)
            deltaW_virginica = self.calculateDeltaW(targets[2], outputs[2], inputs)
            self.weights_setosa += deltaW_setosa
            self.weights_versicolor += deltaW_versicolor
            self.weights_virginica += deltaW_virginica

    def calculateDeltaW(self, target, output, inputs):
        return self.c * (target - output) * np.array(inputs)

    def evaluate_error(self, features, labels):
        errors = 0  #number of errors in epoch
        for itr in range(len(features)):
            target = int(labels[itr][0])
            targets = [0,0,0]
            targets[target] = 1
            inputs = features[itr] + self.bias
            output = self.calculate_output(inputs)
            if output != target:
                errors += 1
        return errors/ len(features)

    def calculate_output(self, inputs, ret_outputs = False):
        net_setosa = np.dot(inputs, self.weights_setosa)
        output_setosa = 1 if net_setosa > 0 else 0
        net_versicolor = np.dot(inputs, self.weights_versicolor)
        output_versicolor = 1 if net_versicolor > 0 else 0
        net_virginica = np.dot(inputs, self.weights_virginica)
        output_virginica = 1 if net_virginica > 0 else 0
        nets = [net_setosa, net_versicolor, net_virginica]
        if ret_outputs:
            return nets.index(max(nets)), [output_setosa, output_versicolor, output_virginica]
        else:
            return nets.index(max(nets))
