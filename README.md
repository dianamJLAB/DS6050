# DS6050: discGAN: distributed, tabular GAN

Repository contains all files related to our DS6050 project on generating healthcare data using our distributed, conditional GAN (discGAN) as well as files associated with distributing CTGAN

Main folder structure of interest:

1) **eICU_gan**: 

**contains files associated with the discGAN implementation**
Folders include:
- data: contains eICU datasets, including different cuts of data we created for our experiments
- data_evaluation: contains our discGAN evaluation notebooks and output
- dataprep: precessing files for the eICU data
- discGAN_models: TF models output for discGAN at different number of epochs and with/without distribution
- dist_output: discGAN generated data results with distribution
- gan_work: *our main GAN files, in which the discGAN Jupyter NB is located*
- non_dist_output: discGAN generated data results without distribution

2) **CTGAN_distribution**:

**contains code for distributing CTGAN, and evaluating both distributed and single-GPU CTGAN**

- ctgan/synthesizers/ctgan.py was updated.
    - the DataParallel library from torch.nn is used
    - CTGANSynthesizer class added two additional arguments:
    - distribute_g: indicator to distribute the generator. This will only work if more than 1 GPU has been provisioned.
    - distribute_d: indicator to distribute the discriminator. This will only work if more than 1 GPU has been provisioned.
    - if more than one GPU is provisions, AND the distribute flag has been set for the generator and/or discriminator, then the generator and/or discriminator is set the DataParallel version of the model.
- test_CTGAN_eICU.ipynb, eval_figs directory, examples/csv/CTGAN_eval.csv, and examples/csv/generated directory were created to evaluate CTGAN on the eICU dataset

