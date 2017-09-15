"""This module includes utilities to run cross-validation on general supervised learning methods."""
from __future__ import division
import numpy as np


def cross_validate(trainer, predictor, all_data, all_labels, folds, params):
    """Perform cross validation with random splits.

    :param trainer: function that trains a model from data with the template
             model = function(all_data, all_labels, params)
    :type trainer: function
    :param predictor: function that predicts a label from a single data point
                label = function(data, model)
    :type predictor: function
    :param all_data: d x n data matrix
    :type all_data: numpy ndarray
    :param all_labels: n x 1 label vector
    :type all_labels: numpy array
    :param folds: number of folds to run of validation
    :type folds: int
    :param params: auxiliary variables for training algorithm (e.g., regularization parameters)
    :type params: dict
    :return: tuple containing the average score and the learned models from each fold
    :rtype: tuple
    """
    scores = np.zeros(folds)#will hold score for each fold

    d, n = all_data.shape #d=rows, n=column

    indices = np.array(range(n), dtype=int)

    # pad indices to make it divide evenly by folds
    examples_per_fold = int(np.ceil(n / folds))
    ideal_length = int(examples_per_fold * folds)
    # use -1 as an indicator of an invalid index
    indices = np.append(indices, -np.ones(ideal_length - indices.size, dtype=int))
    assert indices.size == ideal_length

    indices = indices.reshape((examples_per_fold, folds))

    models = []

    row, col = indices.shape
    
    for i in range(col):
        #switch column at index 0 with column to hold for tests
        indices[:,[0,i]] = indices[:, [i,0]]
        #test indices
        test_indices = indices[:, 0]
        #training indices
        train_indices = indices[:, 1:col].ravel()
        #switch columns back to their original positions
        indices[:,[0,i]] = indices[:, [i,0]]
        #extract test data 
        test_data = all_data[:, test_indices]
        #extract test labels
        test_labels = all_labels[test_indices]
        #extract train data 
        train_data = all_data[:, train_indices]
        #extract train labels
        train_labels = all_labels[train_indices]
        #train on train data and train labels and return a model
        model = trainer(train_data, train_labels, params)
        #append model to models array
        models.append(model)
        #compute labels using predictor function
        labels = predictor(test_data, model)
        #compute accuracy of labels returned by model
        dt_accuracy = np.mean(labels == test_labels)
        #append accuracy to list of scores
        scores[i] = dt_accuracy
        
    score = np.mean(scores)

    return score, models
