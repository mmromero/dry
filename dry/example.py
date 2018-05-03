from correction import Dryann

if __name__ == "__main__":
    bfile = 'tests/test_ok.bvals'
    dwi = 'data/test.nii.gz'
    dann = Dryann()
    model = dann.train_model(bfile)
    dann.correct_fwe(dwi, model, output_folder='tests')
