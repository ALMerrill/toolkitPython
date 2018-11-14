from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
from math import log
import numpy as np
from copy import deepcopy
from scipy.spatial import distance
import matplotlib.pyplot as plt
import sys

class NearestNeighborLearner(SupervisedLearner):
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
        self.weighted = False
        if 'weighted' in sys.argv:
            self.weighted = True

        self.k = 5
        self.training_set = np.array(features.data)
        self.max = self.training_set.max()
        self.training_set /= self.max #normalize between 0 and 1
        self.training_labels = np.array(labels.data)

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        features /= self.max
        isNominal = (self.training_labels.astype(int) == self.training_labels).all()
        labels += [self.determineOutput(features, isNominal)]
    
    def determineOutput(self, features, isNominal):
        distances = np.sqrt(((self.training_set - features)**2).sum(axis=1)) # Euclidean
        # distances = abs((self.training_set - features).sum(axis=1)) # Manhattan
        # distances = abs(self.training_set - features).sum(axis=1) / abs(self.training_set + features).sum(axis=1) # Bray-Curtis
        # distances =  (abs(self.training_set - features) / abs(self.training_set) + abs(features)).sum(axis=1) # Canberra
        # distances = []
        # for instance in self.training_set:
        #     distance = np.sum(np.sqrt((instance - features)**2)) # Euclidean
            # distance = np.sum(abs(instance - features)) # Manhattan
            # distance = np.max(abs(instance - features)) #Chebyshev
            # distance = np.sum(abs(instance - features)) / np.sum(abs(instance + features)) #Bray-Curtis
            # distance = np.sum(abs(instance - features) / abs(instance) + abs(features))
            # mag_inst = np.linalg.norm(instance - np.mean(instance))
            # mag_feat = np.linalg.norm(features - np.mean(features))
            # distance = 1 - (np.dot(instance - np.mean(instance), features - np.mean(features)) / (mag_inst * mag_feat)) # Correlation
            # distance = 1 - (np.dot(instance, features) / (mag_inst * mag_feat))
            # distances.append(distance)
        if 0 in distances:
            distances[distances == 0] = float('inf')
        distances = np.array(distances)
        idx = distances.argsort()[:self.k]
        votes = self.training_labels[idx,0].astype(int)
        if isNominal:
            if self.weighted:
                output = self.getWeightedVote(distances[idx], votes, True) #distance weighted voting
            else:
                output = np.bincount(votes).argmax()
        else:
            if self.weighted:
                inv_sq_distances = 1/(distances[idx]**2)
                output = np.average(a=votes, weights=inv_sq_distances)
            else:
                output = np.mean(votes)
        return output
    
    def distance(self, features, instance):
        if len(features) != len(instance):
            raise ValueError('Feature length and instance length do not match')
        return distance.euclidean(features, instance)
        # dist = 0
        # for i in range(len(features)):
        #     dist += abs(features[i] - instance[i])
        # return dist

    def getWeightedVote(self, distances, votes, isNominal):
        weightedVotes = {}
        for i in range(len(votes)):
            if votes[i] in weightedVotes:
                weightedVotes[votes[i]] += 1/(distances[i]**2)
            else:
                weightedVotes[votes[i]] = 1/(distances[i]**2)
        if isNominal:
            max_vote = -1
            max_weight = -1
            for vote in weightedVotes:
                if weightedVotes[vote] > max_weight:
                    max_weight = weightedVotes[vote]
                    max_vote = vote
            return max_vote
        else:
            print(len(weightedVotes))
            print(weightedVotes.values())
            return np.mean(weightedVotes.values())
