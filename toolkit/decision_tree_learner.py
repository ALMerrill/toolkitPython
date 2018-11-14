from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
from math import log
import numpy as np
from copy import deepcopy

class Node(object):

    def __init__(self, name, children = [], split = None, 
                    output = None, majority = None, isLeaf = False):
        self.name = name
        self.children = children
        self.split = split
        self.output = output
        self.majority = majority
        self.isLeaf = isLeaf

    def __str__(self, level=0):
        ret = "\t"*level + self.name
        if self.output != None:
            ret += " - OUT: " + str(self.output)
        if self.split:
            ret += " - SPLIT: " + self.split['name']
        if self.output == None and self.split == None:
            ret += " None ERROR"
        ret += "\n"
        if self.isLeaf:
            return ret
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return '<tree node representation>'

    def setOutput(self, output):
        self.output = output
        if output is None:
            self.isLeaf = False
        else:
            self.isLeaf = True

    def addChild(self, child):
        self.children.append(child) 

    def hasChildren(self):
        return len(self.children) > 0


class DecisionTreeLearner(SupervisedLearner):
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
        self.labels = []
        for i in range(labels.cols):
            if labels.value_count(i) == 0:
                self.labels += [labels.column_mean(i)]          # continuous
            else:
                self.labels += [labels.most_common_value(i)]    # nominal

        # NUM_BINS = 4
        # for attr in feature_attributes:
        #     for bin_num in range(NUM_BINS):
        #         attr[bin_num] = bin_num
        # features.data = self.binify(np.array(features.data), NUM_BINS) - 1

        self.features_matrix = deepcopy(features)
        self.labels_matrix = deepcopy(labels)
        self.root = Node('root')
        self.feature_names = self.features_matrix.attr_names
        feature_names = deepcopy(self.features_matrix.attr_names)
        feature_attributes = self.features_matrix.enum_to_str
        output_name_map = labels.enum_to_str[0]

        num_rows = features.rows
        training_set = np.array(features.data[:int(num_rows*.8)])
        training_labels = np.array(labels.data[:int(num_rows*.8)])
        validation_set = deepcopy(features)
        validation_labels = deepcopy(labels)
        validation_set.data = np.array(features.data[int(num_rows*.8):])
        validation_labels.data = np.array(labels.data[int(num_rows*.8):])

        # training_set = np.array(features.data)
        # training_labels = np.array(labels.data)

        info = self.info(training_labels, output_name_map)
        self.addToTree(self.root, info, training_set, training_labels, feature_names, output_name_map, feature_attributes)

        self.prune(self.root)

        # print(self.root)
        # print("Node Count:",self.countNodes(self.root))
        # print("Height",self.height(self.root))

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
        
    def addToTree(self, node, info, features, labels, feature_names, output_name_map, feature_attributes):
        node.majority = self.majority(labels)
        if self.isLeaf(labels):
            node.setOutput(labels[0][0])
            return
        elif len(feature_names) <= 1:
            node.setOutput(node.majority)
            return
        sub_infos = []
        sub_entropies = []
        for i in range(len(feature_names)):
            sub_info = self.info_A(features, labels, i, output_name_map, sub_entropies)
            if sub_info < 0:
                raise ValueError('Found negative info value.')
            sub_infos.append(sub_info)
        min_info_index = np.argmin(sub_infos)
        node.split = {}
        node.split['name'] = feature_names[min_info_index]
        node.split['index'] = min_info_index
        indeces = []
        for i in range(len(sub_infos)):
            if i != min_info_index:
                indeces.append(i)
        attr_nodes = self.addAttrNodes(node, features, feature_names, feature_attributes)

        subsets = self.getSubsetsOfData(node, features, labels, feature_names, feature_attributes)
        for i in range(len(subsets)):
            next_labels = subsets[i]['labels']
            next_node = attr_nodes[i]
            next_info = sub_entropies[min_info_index][i]
            next_feature_attributes = feature_attributes[:min_info_index] + feature_attributes[min_info_index+1:]
            next_features = subsets[i]['features'][:,np.array(indeces)]
            next_feature_names = feature_names[:min_info_index] + feature_names[min_info_index+1:]
            self.addToTree(next_node, next_info, next_features, next_labels, next_feature_names, output_name_map, next_feature_attributes)
        
    def majority(self, labels):
        vals = {}
        for i in range(len(labels)):
            val = labels[i][0]
            if val in vals:
                vals[val] += 1
            else:
                vals[val] = 1
        maxVal = 0
        maxKey = -1
        for key in vals:
            if vals[key] > maxVal:
                maxVal = vals[key]
                maxKey = key
        return maxKey
    
    def isLeaf(self, labels):
        return all(i == labels[0] for i in labels)

    def assignOutputs(self, nodes, features, labels):
        ret_outputs = [0 for _ in range(len(nodes))]
        for i in range(len(nodes)):
            outputs = [0 for _ in range(len(nodes))]
            node = nodes[i]
            for instance in features[:,0]:
                if instance == i:
                    outputs[i] += 1
            ret_outputs[i] = np.argmax(outputs)
        for i in range(len(nodes)):
            nodes[i].setOutput(ret_outputs[i])

    def info(self, labels, output_name_map):
        output_map = {}
        num_instances = len(labels)
        for i in range(num_instances):
            output = int(labels[i][0])
            output = output_name_map[output]
            if output in output_map:
                output_map[output] += 1
            else:
                output_map[output] = 1
        info = 0
        for key in output_map:
            fraction = output_map[key] / num_instances
            info += -fraction * log(fraction, 2)
        return info

    def info_A(self, features, labels, feature, output_name_map, sub_entropies):
        attr_map = self.get_attr_map(features, feature)
        info = 0
        entropy = 0
        sub_entropies.append([])
        for attr in attr_map:
            fraction = attr_map[attr] / len(features)
            indeces = self.get_attr_indeces(features, labels, feature, attr)
            #extract a list of labels that correspond to the current key in features
            attr_entropy = self.info(labels[indeces], output_name_map)
            sub_entropies[-1].append(attr_entropy)
            entropy += attr_entropy
            info += attr_entropy * fraction
        return info
    
    def get_attr_indeces(self, features, labels, feature, cat):
        indeces = []
        for i in range(len(features)):
            if features[i][feature] == cat:
                indeces.append(i)
        return np.array(indeces)

    def get_attr_map(self, features, feature):
        attr_map = {}
        num_instances = len(features)
        for i in range(num_instances):
            attribute = int(features[i][feature])
            if attribute in attr_map:
                attr_map[attribute] += 1
            else:
                attr_map[attribute] = 1
        return attr_map

    def getSubsetsOfData(self, node, features, labels, feature_names, feature_attributes):
        feature_index = node.split['index']
        attr_names = feature_attributes[feature_index]
        attr_map = self.get_attr_map(features, feature_index)
        attr_instances = features[:,feature_index]
        subsets = []
        for attr in sorted(attr_map):
            indeces = []
            for i in range(len(attr_instances)):
                if attr_instances[i] == attr:
                    indeces.append(i)
            subsets.append({
                'features': features[np.array(indeces)],
                'labels': labels[np.array(indeces)]
                })
        return subsets
    
    def addAttrNodes(self, node, features, feature_names, feature_attributes):
        feature_index = node.split['index']
        attr_names = feature_attributes[feature_index]
        node.children = [Node(name) for name in attr_names.values()]
        return node.children

    def countNodes(self, node):
        if node.output != None: 
            return 1
        else:
            for i in range(len(node.children)):
                return 1 + sum([self.countNodes(node) for node in node.children])
        return 0

    def height(self, node): 
        if node.isLeaf:
            return 0
        elif node.hasChildren():
            heights = []
            for child in node.children:
                heights.append(self.height(child))
            return max(heights) + 1
        else:
            return 0
    
    def prune(self, root):
        h = self.height(root); 
        for i in reversed(range(h)):
            self.pruneGivenLevel(root, i)

    def pruneGivenLevel(self, root, level): 
        if root.isLeaf:
            return
        if level == 1:
            self.pruneNode(root)
        elif level > 1:
            for child in root.children:
                self.pruneGivenLevel(child, level-1)

    def pruneNode(self, node):
        node.setOutput(node.majority)
        accuracy = self.measure_accuracy(self.features_matrix, self.labels_matrix, Matrix())
        if accuracy > self.maxAccuracy:
            self.maxAccuracy = accuracy
        else:
            node.setOutput(None)

    def binify(self, features, num_bins):
        binned_features = np.zeros(features.shape)
        for i in range(features.shape[1]):
            feature_data = features[:,i]
            max_val = np.max(feature_data)
            min_val = np.min(feature_data)
            bins = np.linspace(min_val, max_val, num_bins)
            binned_features[:,i] = np.digitize(feature_data, bins)
        return binned_features



