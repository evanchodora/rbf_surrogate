import argparse
import numpy as np
import shelve


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
            return np.sqrt((a.T - b) ** 2)
        else:
            return np.sqrt((a.T - a) ** 2)

    def _compute_N(self):

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

        r = self._compute_r(self.x_data)  # Compute the euclidean distance matrix
        return rbf_dict[self.rbf_func](r)

    # Function to train an RBF surrogate using the suplied data and options
    def _train(self):
        N = self._compute_N()  # Compute the basis function matrix of the specified type
        self.weights = np.linalg.solve(N, self.y_data)  # Solve for the weights vector

    def _predict(self):
        N = self._compute_N()

    # Initialization for the RBF class
    def __init__(self, type, x_file, y_file, model_db, rbf_func):

        self.x_data = np.atleast_2d(np.loadtxt(x_file, skiprows=1, delimiter=","))  # Read the input locations file
        self.rbf_func = rbf_func
        self.model_db = model_db

        # Check for training or prediction
        if type == 'train':
            self.y_data = np.atleast_2d(np.loadtxt(y_file, skiprows=1, delimiter=","))  # Read output data file

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
            self.rbf_func = db['rbf_func']
            self.x_train = db['x_train']
            self.weights = db['weights']
            db.close()

            self._predict()


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
