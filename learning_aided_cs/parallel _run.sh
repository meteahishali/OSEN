#!/bin/bash
#
# At first let's give job some descriptive name to distinct
# from other jobs running on cluster
#SBATCH -J example
#SBATCH --array=0-2047
#
# Let's redirect job's out some other file than default slurm-%jobid-out
#SBATCH --output=out_txt/0.05/%a_o1_q3_1.txt
#SBATCH --error=out_txt/0.05/e_%a_o1_q3_1.txt
#
# We'll want to allocate one CPU core
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
#SBATCH --mem=4000
#SBATCH --time=03:59:00
#SBATCH --partition=test
#

module load local-anaconda
conda activate tensorflow_2.4.1

python main_mri_tv.py --learn_aid True --model_aid OSEN1 --sample $SLURM_ARRAY_TASK_ID --mr 0.05 --nu 1 --q 3