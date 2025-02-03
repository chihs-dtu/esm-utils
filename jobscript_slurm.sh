#!/bin/bash
# Job name:
#SBATCH --job-name=protein_encode
#
# GPU allocation
##SBATCH --gres=shard:5
#SBATCH --gres=gpu
#
# Request one node:
#SBATCH --nodes=1
#
# Specifice node:
#SBATCH --nodelist=compute02

# Specify memory for the job (example):
##SBATCH --mem=5G
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
# File for output, use file or /dev/null
#SBATCH -o log/gpu_%J.out
#SBATCH -e log/gpu_%J.err

# Set cuda
export CUDA_VISIBLE_DEVICES=1

# Activate venv
#source /home/people/chihs/protein_embed/esm2_utilities/env/bin/activate

# run training
/home/people/chihs/miniconda3/envs/esm2/bin/python run.py /home/people/chihs/protein_embed/esm2_utilities/chunxu_data/filtered_isoforms_AA.fasta
