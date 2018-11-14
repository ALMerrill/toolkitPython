from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
from math import log
import numpy as np
from copy import deepcopy

class ClusteringLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    labels = []

    def __init__(self):
        self.maxAccuracy = 0
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """

        self.features_matrix = deepcopy(features)
        self.labels_matrix = deepcopy(labels)
        self.root = Node('root')
        self.feature_names = self.features_matrix.attr_names
        feature_names = deepcopy(self.features_matrix.attr_names)
        feature_attributes = self.features_matrix.enum_to_str
        output_name_map = labels.enum_to_str[0]

        # num_rows = features.rows
        # training_set = np.array(features.data[:int(num_rows*.8)])
        # training_labels = np.array(labels.data[:int(num_rows*.8)])
        # validation_set = deepcopy(features)
        # validation_labels = deepcopy(labels)
        # validation_set.data = np.array(features.data[int(num_rows*.8):])
        # validation_labels.data = np.array(labels.data[int(num_rows*.8):])

        training_set = np.array(features.data)
        training_labels = np.array(labels.data)

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        labels += [self.determineOutput(self.root, features)]
    
    def determineOutput(self, node, features):
        if node.isLeaf:
            return node.output
        elif node.split != None:
            splitIndex = node.split['index']
            decisionIndex = int(features[splitIndex])
            return self.determineOutput(node.children[decisionIndex], features[:splitIndex] + features[splitIndex+1:])
        else:
            return 0