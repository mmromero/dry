'''
Created on 6 July 2018
@author: Miguel Molina Romero, Techical University of Munich
@contact: miguel.molina@tum.de
@license: LPGL
'''

import unittest
from dry.dry import Dry
from keras.models import Sequential
import os
import shutil

testingdir = "dry/data/test"
bval1s = "dry/data/one-shell/test_ok.bvals"
dwis1s = ['dry/data/one-shell/test1.nii.gz',
          'dry/data/one-shell/test2.nii.gz',
          'dry/data/one-shell/test3.nii.gz']
bval2s = "dry/data/two-shell/merged.bval"
dwis2s = ["dry/data/two-shell/merged1.nii.gz",
          "dry/data/two-shell/merged2.nii.gz"]
ftissue = "dry/data/two-shell/vftissue.nii.gz"


class TestDry(unittest.TestCase):

    def setUp(self):
        os.makedirs(testingdir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(testingdir, ignore_errors=True)
        shutil.rmtree("test1", ignore_errors=True)
        shutil.rmtree("test2", ignore_errors=True)
        shutil.rmtree("test3", ignore_errors=True)
        pass

    def testTrainModelEmptyBval(self):
        dann = Dry()
        with self.assertRaises(FileNotFoundError):
            dann.train_model("")

    def testTrainModelOk(self):
        dann = Dry()
        model = dann.train_model(bval1s)
        self.assertIsInstance(model, Sequential)

    def testSaveModelNoModel(self):
        dann = Dry()
        with self.assertRaises(ValueError):
            dann.save_model(None, "dry/data/")

    def testSaveModelOK(self):
        dann = Dry()
        model = dann.train_model(bval1s)
        dann.save_model(model, os.path.join(testingdir, "testmodel"))
        self.assertTrue(os.path.isfile(os.path.join(testingdir, "testmodel")))

    def testLoadModelOK(self):
        dann = Dry()
        model = dann.train_model(bval1s)
        dann.save_model(model, os.path.join(testingdir, "testmodel"))
        model2 = dann.load_model(os.path.join(testingdir, "testmodel"))
        self.assertIsInstance(model2, Sequential)

    def testCorrectEmtpyDwi(self):
        dann = Dry()
        model = dann.train_model(bval1s)
        with self.assertRaises(ValueError):
            dann.fwe(None, model, bval1s, testingdir)

    def testCorrectEmptyModel(self):
        dann = Dry()
        with self.assertRaises(ValueError):
            dann.fwe(dwis1s, None, bval1s, testingdir)

    def testCorrectEmptyBval(self):
        dann = Dry()
        model = dann.train_model(bval1s)
        with self.assertRaises(ValueError):
            dann.fwe(dwis1s, model, None, testingdir)

    def testCorrectDifferentBval(self):
        dann = Dry()
        model = dann.train_model(bval1s)
        with self.assertRaises(ValueError):
            dann.fwe(dwis1s, model, bval2s, testingdir)

    def testCorrectEmptyOutputFolder(self):
        dann = Dry()
        model = dann.train_model(bval1s)
        dann.fwe(dwis1s, model, bval1s)
        self.assertTrue(os.path.isfile("test1/tissue_volume_fraction.nii.gz"))
        self.assertTrue(os.path.isfile("test1/fwe_dwi.nii.gz"))
        self.assertTrue(os.path.isfile("test2/tissue_volume_fraction.nii.gz"))
        self.assertTrue(os.path.isfile("test2/fwe_dwi.nii.gz"))
        self.assertTrue(os.path.isfile("test3/tissue_volume_fraction.nii.gz"))
        self.assertTrue(os.path.isfile("test3/fwe_dwi.nii.gz"))

    def testCorrectOneShellOk(self):
        dann = Dry()
        model1s = dann.train_model(bval1s)
        dann.fwe(dwis1s, model1s, bval1s, testingdir)
        self.assertTrue(os.path.isfile(os.path.join(testingdir, "test1/tissue_volume_fraction.nii.gz")))
        self.assertTrue(os.path.isfile(os.path.join(testingdir, "test1/fwe_dwi.nii.gz")))
        self.assertTrue(os.path.isfile(os.path.join(testingdir, "test2/tissue_volume_fraction.nii.gz")))
        self.assertTrue(os.path.isfile(os.path.join(testingdir, "test2/fwe_dwi.nii.gz")))
        self.assertTrue(os.path.isfile(os.path.join(testingdir, "test3/tissue_volume_fraction.nii.gz")))
        self.assertTrue(os.path.isfile(os.path.join(testingdir, "test3/fwe_dwi.nii.gz")))

    def testCorrectTwoShellOk(self):
        dann = Dry()
        model2s = dann.train_model(bval2s)
        dann.fwe(dwis2s, model2s, bval2s, testingdir)
        self.assertTrue(os.path.isfile(os.path.join(testingdir, "merged1/tissue_volume_fraction.nii.gz")))
        self.assertTrue(os.path.isfile(os.path.join(testingdir, "merged1/fwe_dwi.nii.gz")))
        self.assertTrue(os.path.isfile(os.path.join(testingdir, "merged2/tissue_volume_fraction.nii.gz")))
        self.assertTrue(os.path.isfile(os.path.join(testingdir, "merged2/fwe_dwi.nii.gz")))

    def testCorrectFromFEmptyDwis(self):
        dann = Dry()
        with self.assertRaises(ValueError):
            dann.fwe_tissue(None, ftissue, bval2s)

    def testCorrectFromFEmptyFtissue(self):
        dann = Dry()
        with self.assertRaises(ValueError):
            dann.fwe_tissue(dwis2s, None, bval2s)

    def testCorrectFromFEmptyBfile(self):
        dann = Dry()
        with self.assertRaises(ValueError):
            dann.fwe_tissue(dwis2s, ftissue, None)

    def testCorrectFromTissueVolumeFraction(self):
        dann = Dry()
        dann.fwe_tissue(dwis2s[0], ftissue, bval2s, testingdir)
        self.assertTrue(os.path.isfile(os.path.join(testingdir, "merged1/fwe_dwi.nii.gz")))
