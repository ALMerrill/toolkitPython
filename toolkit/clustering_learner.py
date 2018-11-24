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

        self.k = 5
        self.iters = 7

        self.training_set = np.delete(np.array(features.data),0,axis=1)
        features.attr_names = features.attr_names[1:]
        features.str_to_enum = features.str_to_enum[1:]
        # self.training_set /= self.max
        self.training_labels = np.array(labels.data)

        self.centroids = self.training_set[:5]
        self.clusters = [[self.centroids[i]] for i in range(self.k)]

        for i in range(self.iters):
            print("***************")
            print("Iteration", i+1)
            print("***************")
            print("Computing Centroids:")
            for i in range(len(self.centroids)):
                print("Centroid " + str(i) + " = ", end="")
                for j in range(len(self.centroids[i])):
                    val = self.centroids[i][j]
                    if j == len(self.centroids[i]) - 1:
                        print(val)
                    else:
                        print(val,end=", ")
            print("Making Assignments")
            self.determineClusters(features)
            self.clusters = self.recalculateCentroids()
            assert len(self.clusters) == self.k
            assert type(self.clusters[0]) == list

        # num_rows = features.rows
        # training_set = np.array(features.data[:int(num_rows*.8)])
        # training_labels = np.array(labels.data[:int(num_rows*.8)])
        # validation_set = deepcopy(features)
        # validation_labels = deepcopy(labels)
        # validation_set.data = np.array(features.data[int(num_rows*.8):])
        # validation_labels.data = np.array(labels.data[int(num_rows*.8):])

        training_set = np.array(features.data)
        training_labels = np.array(labels.data)

    def determineClusters(self, features):
        sse = [0 for _ in range(self.k)]
        for i, instance in enumerate(self.training_set):
            min_dist = float('inf')
            closest_cluster = 0
            isCentroid = False
            for j, centroid in enumerate(self.centroids):
                if np.array_equal(instance, centroid):
                    print(str(i) + "=" + str(i), end=" ")
                    isCentroid = True
                    break
                distance = self.calculateDistance(instance, centroid, features)
                if distance < min_dist:
                    min_dist = distance
                    closest_cluster = j
            if isCentroid:
                continue
            sse[closest_cluster] += min_dist**2
            self.clusters[closest_cluster].append(instance)
            print(str(i) + "=" + str(closest_cluster), end="  ")
            if i % 10 == 9:
                print()
        print("\n"+ "SSE: " + str(sum(sse)))
        print()

    def calculateDistance(self, instance, centroid, features):
        distance = 0
        for j in range(len(centroid)):
            x = instance[j]
            y = centroid[j]
            if self.isNominal(j, features):
                if x == -1 or y == -1:
                    distance += 1
                else:
                    distance += 1 if x != y else 0
            else:
                if x == -1 or y == -1:
                    distance += 1
                else:
                    distance += (x - y)**2
        return distance

    def isNominal(self, index, features):
        return bool(features.str_to_enum[index])

    def recalculateCentroids(self):
        # centroids = np.mean(self.centroids, axis=1).tolist()
        # print(centroids)
        # for centroid in centroids:
        #     centroid.tolist()
        new_centroids = [[] for _ in range(self.k)]
        for i, cluster in enumerate(self.clusters):
            print('here')
            new_centroids[i] = np.mean(cluster, axis=0)
        return new_centroids


    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        labels += [self.determineOutput()]
    
    def determineOutput(self):
        return 0