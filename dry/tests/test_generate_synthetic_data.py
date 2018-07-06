'''
Created on 3 May 2018
@author: Miguel Molina Romero, Techical University of Munich
@contact: miguel.molina@tum.de
@license: LPGL
'''

import unittest
from dry.utils import generate_synthetic_data, DryException


class TestGenerateSyntheticData(unittest.TestCase):
    def testEmptyBfile(self):
        try:
            generate_synthetic_data(None)
            self.fail('generate_synthetic_data did not raise an excpetion')
        except DryException:
            pass
        except Exception:
            self.fail('Wrong exception raised')

    def testWrongBfileContent(self):
        bfile = 'dry/data/one-shell/test_ko.bvals'
        try:
            generate_synthetic_data(bfile)
            self.fail('generate_synthetic_data did not raise an excpetion')
        except ValueError:
            pass
        except Exception:
            self.fail('Wrong exception raised')

    def testCorrectBfile(self):
        bfile = 'dry/data/one-shell/test_ok.bvals'
        sdata = generate_synthetic_data(bfile)
        self.assertIsNotNone(sdata)
        self.assertTrue(sdata['S'].shape == (50000, 31))
        self.assertTrue(sdata['f'].shape == (50000, 1))
