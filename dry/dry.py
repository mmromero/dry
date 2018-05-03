'''
Created on 1 May 2018

High level class containing the necessary methods.

@author: Miguel Molina Romero, Technical University of Munich
@contact: miguel.molina@tum.de
@License: LPGL
'''


class Dry:
    '''Dry free-water with artificial neural networks (ANN).

    Functions:
        - train_model: trains an ANN model from the given b-values fileself.
        - load_model: loads an existing ANN model from the given file.
        - save_model: saves a new trained model.
        - correct_fwe: Corrects free-water contamination from a list diffusion
                       weighted volumes using the given model.
    '''

    def train_model(self, bfile):
        '''
        '''

    def load_model(self, mfile):
        '''
        '''

    def save_model(self, mfile, model):
        '''
        '''

    def correct_fwe(self, dwi, model, output_folder=None):
        '''
        '''
