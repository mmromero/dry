'''
Created on 1 May 2018

Utility functions.

@author: Miguel Molina Romero, Technical University of Munich
@contact: miguel.molina@tum.de
@License: LPGL
'''


def generate_synthetic_data(bfile):
    '''It generates free-water contaminated synthetic data for training.

    param bfile: b-values file.

    rtype: numpy matrices
    return: X, containing the contaminated signal, and Y, containing the
            free-water volume fraction
    '''
