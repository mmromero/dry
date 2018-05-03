'''
Created on 3 May 2018
@author: Miguel Molina Romero, Techical University of Munich
@contact: miguel.molina@tum.de
@license: LPGL
'''

import unittest
import dry.utils as ut


class TestGenerateSyntheticData(unittest.TestCase):
    def testEmptyBfile(self):
        try:
            ut.generate_synthetic_data(None)
            self.fail('generate_synthetic_data did not raise an excpetion')
        except Exception:
            pass

    def testWrongBfileContent(self):
        bfile = 'test_ko.bvals'
        try:
            ut.generate_synthetic_data(bfile)
            self.fail('generate_synthetic_data did not raise an excpetion')
        except Exception:
            pass

    def testCorrectBfile(self):
        bfile = 'test_ok.bvals'
        sdata = ut.generate_synthetic_data(bfile)
        self.assertTrue(sdata['S'].size is (31, 50000))
        self.assertTrue(sdata['f'].size is (1, 50000))


if __name__ == "__main__":
    unittest.main()
