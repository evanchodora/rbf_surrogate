import argparse
import numpy as np
import shelve
from scipy.spatial.distance import squareform, cdist, pdist


'''
Python Tool for Training RBF Surrogate Models

    Evan Chodora (2019)
    echodor@clemson.edu
'''


# Class to create or use and RBF surrogate model
class RBF:

    # Collection of possible Radial Basis Functions to use in the surrogate model:
    # Multiquadratic
    def _multiquadric(self, r):
        return np.sqrt((1.0/self.epsilon*r)**2 + 1)

    # Inverse Multiquadratic
    def _inverse_multiquadric(self, r):
        return 1.0/np.sqrt((1.0/self.epsilon*r)**2 + 1)

    # Standard Gaussian
    def _gaussian(self, r):
        return np.exp(-(1.0/self.epsilon*r)**2)

    # Linear
    def _linear(self, r):
        return r

    # Cubic
    def _cubic(self, r):
        return r**3

    # Thin Plate
    def _thin_plate(self, r):
        return xlogy(r**2, r)

    # Function to compute the Euclidean distance (r)
    def _compute_r(self, a, b=None):
        if b is not None:
            # Return the euclidean distance matrix between two matrices (or vectors)
            return cdist(a, b, 'euclidean')
        else:
            # Return a square matrix form of the the pairwise euclidean distance for the training locations
            return squareform(pdist(a, 'euclidean'))

    def _compute_N(self, r):

        # Dictionary object to store possible RBFs and associated functions to evaluate them
        # Can add as needed when a new function is added to the collection above
        rbf_dict = {
            "multiquadratic" : self._multiquadric,
            "inverse multiquadric" : self._inverse_multiquadric,
            "gaussian" : self._gaussian,
            "linear" : self._linear,
            "cubic" : self._cubic,
            "thin plate" : self._thin_plate
        }

        return rbf_dict[self.rbf_func](r)

    # Function to train an RBF surrogate using the suplied data and options
    def _train(self):
        r = self._compute_r(self.x_data)  # Compute the euclidean distance matrix
        N = self._compute_N(r)  # Compute the basis function matrix of the specified type

        self.weights = np.linalg.solve(N, self.y_data)  # Solve for the weights vector

    def _predict(self):
        r = self._compute_r(self.x_train, self.x_data)
        N = self._compute_N(r)
        self.y_pred = np.dot(N.T, self.weights)

    # Initialization for the RBF class
    def __init__(self, type, x_file, y_file, model_db, rbf_func):

        self.x_data = np.loadtxt(x_file, skiprows=1, delimiter=",")  # Read the input locations file
        self.x_data = self.x_data.reshape(self.x_data.shape[0], -1)
        self.rbf_func = rbf_func
        self.model_db = model_db

        # Check for training or prediction
        if type == 'train':
            self.y_data = np.loadtxt(y_file, skiprows=1, delimiter=",")  # Read output data file
            self.y_data = self.y_data.reshape(self.y_data.shape[0], -1)
            self._train()  # Run the model training function

            # Store model parameters in a python shelve database
            db = shelve.open(self.model_db)
            db['rbf_func'] = self.rbf_func
            db['x_train'] = self.x_data
            db['weights'] = self.weights
            db.close()

        else:
            # Read previously stored model data
            model_data = shelve.open(model_db)  # Otherwise, read stored model data
            self.rbf_func = model_data['rbf_func']
            self.x_train = model_data['x_train']
            self.weights = model_data['weights']
            model_data.close()

            self._predict()  # Run the model prediction functions

            # Quick loop to add a header like the input files
            y_head = []
            for i in range(self.y_pred.shape[1]):
                y_head.append('y' + str(i))

            # Convert header list to string and write out the predictions to a file
            header = ','.join(y_head)
            np.savetxt('y_pred.dat', self.y_pred, delimiter=',', fmt="%.6f", header=header, comments='')


# Code to run when called from the command line (usual behavior)
if __name__ == "__main__":

    # Parse the command line input options to "opt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--type', dest='type', choices=['train', 'predict'], required=True,
                        help="""Specify whether the tool is to be used for training with \"train\" or
                        making predictions with a stored model with \"predict\".""")
    parser.add_argument('-x', dest='x_file', default='x_train.dat',
                        help="""Input file of x locations. Default is \"x_train.dat\".""")
    parser.add_argument('-y', dest='y_file', default='y_train.dat',
                        help="""Output file for surrogate training. Default is \"y_train.dat\".""")
    parser.add_argument('-m', '--model', dest='model', default='model.db',
                        help="""File to save the model output or use a previously trained model file.
                        Default is \"model.db\".""")
    parser.add_argument('-b', '--rbf', dest='rbf', default='gaussian',
                        help="""Specified RBF to use when training the surrogate. Default is \"gaussian\".""")
    opts = parser.parse_args()

    surrogate = RBF(opts.type, opts.x_file, opts.y_file, opts.model, opts.rbf)
