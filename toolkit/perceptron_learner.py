from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix

import random
import numpy as np


class PerceptronLearner(SupervisedLearner):
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
        self.c = .1
        self.bias = [1]
        self.weights = [0 for _ in range(features.cols + 1)]
        output = 0

        MAX_EPOCHS = 1000
        STOP_EPOCHS = 6
        ERROR_BOUND = .01
        total_epochs = 0
        epochs = 1
        err = 1
        least_error = 1
        least_weights = []
        for epoch in range(MAX_EPOCHS):
            total_epochs += 1
            features.shuffle(labels)
            self.one_training_epoch(features, labels)
            error_rate = self.evaluate_error(features, labels)
            if error_rate < least_error:
                least_error = error_rate
                least_weights = self.weights
            if abs(error_rate - err) < ERROR_BOUND:
                epochs += 1
                if epochs == STOP_EPOCHS:
                    if error_rate > least_error:
                        error_rate = least_error
                        self.weights = least_weights
                    break
            else:
                epochs = 1
                err = error_rate

            # print(total_epochs, end=", ")
            print(error_rate)
        print(self.weights)


    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        target = labels
        net = 0
        inputs = features + self.bias
        for i in range(len(inputs)):
            net += inputs[i] * self.weights[i]
        output = 1 if net > 0 else 0
        prediction = [output]
        labels += prediction


    def one_training_epoch(self, features, labels):
        errors = 0  #number of errors in epoch
        for itr in range(features.rows):
            target = labels.row(itr)[0]
            net = 0
            inputs = features.row(itr) + self.bias
            net = np.dot(inputs, self.weights)
            output = 1 if net > 0 else 0
            if output != target:
                errors += 1
            deltaW = self.c * (target - output) * np.array(inputs)
            self.weights += deltaW

    def evaluate_error(self, features, labels):
        errors = 0  #number of errors in epoch
        for itr in range(features.rows):
            target = labels.row(itr)[0]
            net = 0
            inputs = features.row(itr) + self.bias
            net = np.dot(inputs, self.weights)
            output = 1 if net > 0 else 0
            if output != target:
                errors += 1
        return errors/ features.rows