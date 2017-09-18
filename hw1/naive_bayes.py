"""This module includes methods for training and predicting using naive Bayes."""
from __future__ import division
import numpy as np

def naive_bayes(train_data, train_labels, num_classes, alpha):
    
    labels = np.unique(train_labels)
    number_of_classes = np.zeros(num_classes) # [0,0,0,0....] length = 20
    #shape of data
    d,n = train_data.shape
    # probability of calsses 
    class_prob = np.zeros(num_classes)
    # this will hold a P array of probability for each feature given a specific class 
    class_prob_of_x_given_y =  np.empty((0,d))
    
    for c in range(num_classes):
        number_of_classes[c] = np.sum(train_labels == c)
        # compute probability for a class (adding alpha to avoid division by zero)
        class_prob[c] = (number_of_classes[c] + alpha) / (n + 2*alpha)
        # this next line sums across the rows of data, multiplied by the
        # indicator of whether each column's label is c. It counts the number
        # of times each feature is on among examples with label c
        num_y_and_x = np.asarray(train_data[:, train_labels == c].sum(1)).T
        #probability of a feature given class 
        prob_of_x_given_y = (num_y_and_x + alpha)/ (number_of_classes[c] + 2*alpha)
        # store this p vector inside its class array
        cur_matrix = ((prob_of_x_given_y))
        # sum up all the value for each feature
        #cur_matrix = cur_matrix.sum()
        row_for_one_class = (cur_matrix * (class_prob[c]))
        # add the log of the probability
        #print(" row for one class ", row_for_one_class.shape)
        #print(class_prob_of_x_given_y.shape)        
        #print ("c is ", c , "number of classes", number_of_classes[c], "probability is ", class_prob[c])
        class_prob_of_x_given_y = np.append(class_prob_of_x_given_y, row_for_one_class, axis=0)
        
    return class_prob_of_x_given_y
    
        
def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include an 'alpha' value
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """
    alpha = params['alpha']
    
    labels = np.unique(train_labels)
    
    d, n = train_data.shape
    num_classes = labels.size

    # TODO: INSERT YOUR CODE HERE TO LEARN THE PARAMETERS FOR NAIVE BAYES
    #print (params)
    model = naive_bayes(train_data, train_labels, num_classes, alpha)
    print(model)
    return (model)
                                      
def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    # TODO: INSERT YOUR CODE HERE FOR USING THE LEARNED NAIVE BAYES PARAMETERS
    # TO CLASSIFY THE DATA
    # d is the number of feature and  20x5000
    c,d = model.shape
    result = np.zeros(c)
    data = data.T
    (n, k) = data.shape
    matrix_of_class_val =  np.empty((0,n))
    #not_data = data[data != True]
    for i in range (c):
        new_data = data.copy()
        new_data[new_data == False]= -1
        new_data[new_data == True]= 0
        current_row = model[i,:]
        class_val = current_row + new_data
        class_val[class_val < 0] *= -1
        # this should be 11,xxx by 5000
        #print (class_val.shape)
        class_val = np.log2(class_val)
        # add rows across
        class_val = (class_val.sum(axis=1)).T
        #this should be an array of 11,xxx
        #print(":after")
        #print (class_val.shape)
        #print(matrix_of_class_val.shape)
        matrix_of_class_val = np.append(matrix_of_class_val, class_val, axis=0)
        
    result = (matrix_of_class_val.argmax(axis=0))
    return result
    
    
                                      
                                      
                                      
                                      
                                    
