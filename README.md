# dry
Free-water elimination for diffusion MRI with synthetic data and deep learning.

Diffusion metrics are typically biased by Cerebrospinal fluid (CSF) contamination. In this work, we present a deep learning based solution to remove the CSF contribution. First, we train an artificial neural network (ANN) with synthetic data to estimate the tissue volume fraction. Second, we use the resulting network to predict estimates of the tissue volume fraction for real data, and use them to correct for CSF contamination. 

***
## How to cite
M. Molina-Romero, B. Wiestler, PA. Gómez, MI. Menzel, BH. Menze. Deep learning with synthetic diffusion MRI data for free-water elimination in glioblastoma cases. In: MICCAI: International Conference on Medical Image Computing and Computer-Assisted Intervention (2018).
