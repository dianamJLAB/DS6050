# DS6050

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

2) CTGAN_distribution:
