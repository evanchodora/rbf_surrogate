import argparse
import numpy as np


'''
Python Tool for Training RBF Surrogate Models

    Evan Chodora (2019)
    echodor@clemson.edu
'''


class RBF(object):

    def __init__(self, *args, **kwargs):

        self.x = args[0]  # First argument is 2D array of input values (n * N)
        self.dim = self.x.shape[-1]  # Calculate number of ***
        self.y = args[1]

if __name__ == "__main__":

    # Parse the command line input options
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--type', choices=['train', 'predict'], required=True, help="Specify whether the tool is to be used for training with \"train\" or making predictions with a stored model with \"predict\".")
    parser.add_argument('-x', required=True, help="Input file of x locations")
    parser.add_argument('-y', help="Output file for surrogate training")
    parser.add_argument('-m', '--model', default='model.db', help="File to save the model output or use a previously trained model file. Default is \"model.db\".")
    parser.add_argument('-b', '--rbf', default='gaussian', help="Specified RBF. Default is a Gaussian.")
    opts = parser.parse_args()
