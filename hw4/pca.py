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

    #####################################################################
    # End of your contributed code
    #####################################################################

    return new_data, variances, eigenvectors
