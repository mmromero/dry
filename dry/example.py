from dry.dry import Dry

if __name__ == "__main__":
    bfile = 'dry/data/one-shell/test_ok.bvals'
    dwi = ['dry/data/one-shell/test1.nii.gz',
           'dry/data/one-shell/test2.nii.gz',
           'dry/data/one-shell/test3.nii.gz']
    dann = Dry()
    model1s = dann.train_model(bfile)
    dann.fwe(dwi, model1s, bfile, output_folder='dry/data')

    bfile = 'dry/data/two-shell/merged.bval'
    dwi = ['dry/data/two-shell/merged1.nii.gz',
           'dry/data/two-shell/merged2.nii.gz']
    dann = Dry()
    model2s = dann.train_model(bfile)
    dann.fwe(dwi, model2s, bfile, output_folder='dry/data')
