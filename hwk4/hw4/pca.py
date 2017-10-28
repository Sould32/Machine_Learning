"""This module contains the function to perform principal components analysis."""
import numpy as np


def pca(data):
    """
    Perform principal components analysis to reduce dimensionality of data.

    :param data: d x n matrix of n d-dimensional data points. Each column is an example.
    :type data: ndarray
    :return: tuple containing three components: (new_data, variances, eigenvectors). The variable new_data is a d x n
    matrix containing the original data mapped to a new coordinate space. The variable variances is a length-d vector
    containing the variance captured by each new dimensions. The variable eigenvectors is a matrix where each column
    is one of the eigenvectors that the data has been projected onto.
    :rtype: tuple
    """
    #####################################################################
    # Enter your code below for computing new_data and variances.
    # You may use built in np.linalg.eig or np.linalg.svd, but you are
    # not allowed to use a pre-built pca in your implementation
    #####################################################################
    (d, n) = data.shape
    #print (data)
    mean_data = np.mean(data, 1)
    #print (mean_data)
    mean_data = mean_data.reshape(d,1)
    mean_data_matrix = np.tile(mean_data, (1,n))
    #mean = np.tile(np.mean(data,1).reshape(d,1), (1, n))
    data -= mean_data_matrix
    #print (data)
    variances = np.zeros(n)
    co_variance = np.dot(data, data.T)/n
    print (co_variance)
    variances, eigenvectors = np.linalg.eig(co_variance)
    #ranks = variances.argsort()[::-1]
    #variances = variances[ranks]
    #eigenvectors = eigenvectors[:, ranks]
    new_data = np.dot(eigenvectors.T, data)
    
    #####################################################################
    # End of your contributed code
    #####################################################################

    return np.real(new_data), np.real(variances), np.real(eigenvectors)
