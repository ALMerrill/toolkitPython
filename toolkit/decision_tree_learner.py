from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
from math import log
import numpy as np

class Node(object):

    def __init__(self, name, children = [], output = None):
        self.name = name
        self.children = children
        self.output = output

    def __str__(self, level=0):
        ret = "\t"*level+self.name+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return '<tree node representation>'

    def addChild(self, child):
        self.children.append(child)

# class Tree(object):

#     def __init__(self, root):
#         self.root = root        


class DecisionTreeLearner(SupervisedLearner):
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
        self.labels = []
        for i in range(labels.cols):
            if labels.value_count(i) == 0:
                self.labels += [labels.column_mean(i)]          # continuous
            else:
                self.labels += [labels.most_common_value(i)]    # nominal

        self.features_matrix = features
        self.labels_matrix = labels

        root = Node('root')
        feature_names = self.features_matrix.attr_names
        output_name_map = labels.enum_to_str[0]
        label_data = np.array(labels.data)
        feature_data = np.array(features.data)
        info = self.info(label_data, output_name_map)
        self.addToTree(root, info, feature_data, label_data, feature_names, output_name_map)
        # print(root)

        # print("Info:",info)
        # info_meat = self.info_A(feature_data, label_data, 0, output_name_map)
        # print("Info Meat:",info_meat)
        # info_crust = self.info_A(feature_data, label_data, 1, output_name_map)
        # print("Info Crust:",info_crust)
        # info_veg = self.info_A(feature_data, label_data, 2, output_name_map)
        # print("Info Veg:",info_veg)

    def addToTree(self, node, info, features, labels, feature_names, output_name_map):
        info = self.info(labels, output_name_map)
        sub_infos = []
        node.children = [Node(name) for name in feature_names]
        for i in range(len(feature_names)):
            sub_info = self.info_A(features, labels, i, output_name_map)
            if sub_info < 0:
                raise ValueError('Found negative info value.')
            sub_infos.append(sub_info)
        min_info_index = np.argmin(sub_infos)
        self.addAttrNodes(node.children[min_info_index], features, feature_names)

        self.splitData(node.children[min_info_index], features, labels, feature_names)
        # self.addToTree(node.children[min_info_index], )
        # print(sub_infos)

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
        # total = output_map['total']
        for key in output_map:
            fraction = output_map[key] / num_instances
            info += -fraction * log(fraction, 2)
        return info

    def info_A(self, features, labels, feature, output_name_map):
        attr_map = self.get_attr_map(features, feature)
        info = 0
        for attr in attr_map:
            fraction = attr_map[attr] / len(features)
            indeces = self.get_attr_indeces(features, labels, feature, attr)
            #extract a list of labels that correspond to the current key in features
            info += fraction * self.info(labels[indeces], output_name_map)
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

    def splitData(self, node, features, labels, feature_names):
        # basically needs to return only the instances that correspond with 'meat' 'no' or whatever attr we're on
        print(feature_names)
        feature_index = feature_names.index(node.name)
        attr_names = self.features_matrix.enum_to_str[feature_index]
        attr_map = self.get_attr_map(features, feature_index)
        attr_instances = features[:,feature_index]
        indeces = []
        for attr in attr_map:
            for i in range(len(attr_instances)):
                if attr_instances[i] == attr:
                    indeces.append(i)
            print()
        print(attr_map)
        print(features[:,feature_index])
    
    def addAttrNodes(self, node, features, feature_names):
        feature_index = feature_names.index(node.name)
        attr_names = self.features_matrix.enum_to_str[feature_index]
        node.children = [Node(name) for name in attr_names.values()]

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        labels += self.labels
        # print(labels)



