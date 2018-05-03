'''
Created on 3 May 2018
@author: Miguel Molina Romero, Techical University of Munich
@contact: miguel.molina@tum.de
@license: LPGL
'''

import unittest
from dryann import utils as ut


class TestGenerateSyntheticData(unittest.TestCase):
    def testEmptyBfile(self):
        with self.assertRaises(Exception) as context:
            sdata = ut.generate_synthetic_data(None)
            self.assertIsNone(sdata, 'Unexpected result')
            self.assertTrue('Missing b-file' in context.exception)

    def testWrongBfileContent(self):
        with self.assertRaises(Exception) as context:
            sdata = ut.generate_synthetic_data(None)
            self.assertIsNone(sdata, 'Unexpected result')
            self.assertTrue('Wrong b-file content' in context.exception)

    def testCorrectBfile(self):
        bfile = 'test.bvals'
        sdata = ut.generate_synthetic_data(bfile)
        self.assertTrue(sdata['S'].size is (31, 50000))
        self.assertTrue(sdata['f'].size is (1, 50000))


if __name__ == "__main__":
    unittest.main()
