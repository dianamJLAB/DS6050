# Artificial_Genomes

- Artificial genomes (AGs) created via GAN and RBM models. Genomes from 1000 Genomes Phase 3 have been used for training.
- Genomes from 805 SNP files have 805 highly informative SNPs in terms of population structure.
- Genomes from 10K SNP files have 10000 SNPs from a section of chromosome 15.
- Rows are haplotypes and columns are SNP positions except for the first column which is group tag and the second column which is ID. 
    - Only biallelic SNPs are included with 0 and 1 tags for minor and major allele, respectively.
- You can check the [preprint](https://www.biorxiv.org/content/10.1101/769091v2) for further details.


### Steps For iFarm
1. ssh
2. cd Documents/data_science/PlayGround/UVa/DS6050/artificial_genomes
3. salloc --gres gpu:TitanRTX:1 --partition gpu --nodes 1 --time=12:00:00 --mem=24GB srun --pty bash
4. source /etc/profile.d/modules.sh
5. module use /apps/modulefiles
6. module load anaconda3/4.5.12
7. conda activate gengan
8. python gan_script5.py > OUTPUT_gengan.txt