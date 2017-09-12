"""This module includes methods for training and predicting using decision trees."""
from __future__ import division
import numpy as np
import pandas as pd

def calculate_information_gain(data, labels):
    """Compute the information gain on label probability for each feature in data.

    :param data: d x n matrix of d features and n examples
    :type data: ndarray
    :param labels: n x 1 vector of class labels for n examples
    :type labels: array
    :return: d x 1 vector of information gain for each feature (H(y) - H(y|x_d))
    :rtype: array
    """
    num_classes = len(np.unique(labels))

    class_count = np.zeros(num_classes)

    d, n = data.shape

    full_entropy = 0
    for c in range(num_classes):
        class_count[c] = np.sum(labels == c)
        if class_count[c] > 0:
            class_prob = class_count[c] / n
            full_entropy -= class_prob * np.log(class_prob)

    # print("Full entropy is %d\n" % full_entropy)

    gain = full_entropy * np.ones((1, d))
    num_x = np.asarray(data.sum(1)).T
    prob_x = num_x / n
    prob_not_x = 1 - prob_x

    for c in range(num_classes):
        # print("Computing contribution of class %d." % c)
        num_y = np.sum(labels == c)
        # this next line sums across the rows of data, multiplied by the
        # indicator of whether each column's label is c. It counts the number
        # of times each feature is on among examples with label c
        num_y_and_x = np.asarray(data[:, labels == c].sum(1)).T

        # Prevents Python from outputting a divide-by-zero warning
        with np.errstate(invalid='ignore'):
            prob_y_given_x = num_y_and_x / num_x
        prob_y_given_x[num_x == 0] = 0

        nonzero_entries = prob_y_given_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_x, prob_y_given_x), np.log(prob_y_given_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]

        # The next lines compute the probability of y being c given x = 0 by
        # subtracting the quantities we've already counted
        # num_y - num_y_and_x is the number of examples with label y that
        # don't have each feature, and n - num_x is the number of examples
        # that don't have each feature
        with np.errstate(invalid='ignore'):
            prob_y_given_not_x = (num_y - num_y_and_x) / (n - num_x)
        prob_y_given_not_x[n - num_x == 0] = 0

        nonzero_entries = prob_y_given_not_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_not_x, prob_y_given_not_x), np.log(prob_y_given_not_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]

    return gain.ravel()

def d_all_one_class(data, labels, num_classes):
    
    class_count = np.zeros(num_classes) # [0,0,0,0....] length = 20
    d, n = data.shape # (5000, 11,269)
    most_c = None
    cur_max = -1
    for c in range(num_classes):
        class_count[c] = np.sum(labels == c)
        if (class_count[c] == n):
            most_c = c
            return (True, most_c)
        if(cur_max <= class_count[c]):
            cur_max = class_count[c]
            most_c = c
    return (False, most_c)

    
def rec_tree_train (data, labels, depth, max_depth, num_classes, node, root):
    result, most_c = d_all_one_class(data, labels, num_classes)
    node[root] = {"prediction": None, "l_child": None, "r_child":None}
    if (depth >= max_depth or result):
        node[root] = {"prediction":most_c, "l_child": None, "r_child":None}
        return node
    
    #grabs the boolean values related to the feature at row 0 which represent the feature with the higest gain
    positive_bool_array = data[0,:].toarray().ravel()
    #compute the inverse of that array 
    negative_bool_array = ~positive_bool_array
    #grabs all columns where the feature is true
    left_split = data[:, positive_bool_array]
    #grabs all columns where the feature is false
    right_split = data[:, negative_bool_array]
    #compute information gain for the left split 
    left_labels = labels[positive_bool_array]
    left_rule = calculate_information_gain (left_split, left_labels)
    left_rank = left_rule.argsort()[::-1]
    print("rank of left split")
    print(left_rank[0])
    print("current root")
    print(root)
    #compute information gain for the right split 
    right_labels = labels[negative_bool_array]
    right_rule = calculate_information_gain (right_split, right_labels)
    right_rank = right_rule.argsort()[::-1]
    print("rank of right split")
    print(right_rank[0])
    print("current root")
    print(root)
    #sort data base on feature with the more gain
    data_left = left_split[left_rank[:5000], :]
    data_right = right_split[right_rank[:5000], :]
    #increase depth
    depth += 1
    #recursive compute subtree for left and right child
    node[root]["l_child"]= rec_tree_train(data_left, left_labels, depth, max_depth, num_classes, node, left_rank[0])
    node[root]["r_child"] = rec_tree_train(data_right, right_labels, depth, max_depth, num_classes, node, right_rank[0])
    return node
    
def recursive_tree_train(data, labels, depth, max_depth, num_classes):
    """Helper function to recursively build a decision tree by splitting the data by a feature.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param labels: length n numpy array with integer labels
    :type labels: array_like
    :param depth: current depth of the decision tree node being constructed
    :type depth: int
    :param max_depth: maximum depth to expand the decision tree to
    :type max_depth: int
    :param num_classes: number of classes in the classification problem
    :type num_classes: int
    :return: dictionary encoding the learned decision tree node
    :rtype: dict
    """
    # TODO: INSERT YOUR CODE FOR LEARNING THE DECISION TREE STRUCTURE HERE
    node = dict()
    
    node = rec_tree_train (data, labels, depth, max_depth, num_classes, node, 0)
   
    return node

def decision_tree_train(train_data, train_labels, params):
    """Train a decision tree to classify data using the entropy decision criterion.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include a 'max_depth' value
    :type params: dict
    :return: dictionary encoding the learned decision tree
    :rtype: dict
    """
    max_depth = params['max_depth']

    labels = np.unique(train_labels)
    num_classes = labels.size

    model = recursive_tree_train(train_data, train_labels, depth=0, max_depth=max_depth, num_classes=num_classes)
    return model

def decision_tree_predict(data, model):
    """Predict most likely label given computed decision tree in model.

    :param data: d x n ndarray of d binary features for n examples.
    :type data: ndarray
    :param model: learned decision tree model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    # TODO: INSERT YOUR CODE FOR COMPUTING THE DECISION TREE PREDICTIONS HERE
    print(model)
    labels = {0}
    return labels
