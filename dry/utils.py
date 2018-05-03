'''
Created on 1 May 2018

Utility functions.

@author: Miguel Molina Romero, Technical University of Munich
@contact: miguel.molina@tum.de
@License: LPGL
'''

import numpy as np


class DryException(Exception):
    def __init__(self, message, errors=0):

        # Call the base class constructor with the parameters it needs
        super(DryException, self).__init__(message)

        # Now for your custom code...
        self.errors = errors


def generate_synthetic_data(bfile):
    '''It generates free-water contaminated synthetic data for training.

    param bfile: b-values file.

    rtype: numpy matrices
    return: X, containing the contaminated signal, and Y, containing the
            free-water volume fraction
    '''

    N = 50000

    if bfile is None:
        raise DryException('generate_synthetic_data did \
                            not raise an excpetion')

    bvals = np.loadtxt(bfile, float, delimiter=' ')
    numbs = bvals.size

    Dfw = 3e-3
    Sfw = np.exp(-bvals * Dfw)
    Sfw = np.tile(Sfw, (N, 1))

    ffw = np.random.uniform(size=(N, 1))
    ft = 1 - ffw

    St = np.random.uniform(size=(N, numbs))
    St[:, np.where(bvals == 0)] = 1

    S = np.multiply(St, ft) + np.multiply(Sfw, ffw)

    return {'S': S, 'f': ft}
