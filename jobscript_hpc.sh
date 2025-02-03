#!/bin/sh 
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J protein_coding
### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "select[gpu32gb]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 10GB of system-memory
#BSUB -R "rusage[mem=10GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u chihs@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o log/gpu_%J.out
#BSUB -e log/gpu_%J.err
# -- end of LSF options --


date=$(date +%Y%m%d_%H%M)

# Activate venv
source ../env/bin/activate

# run training
python3 run.py chunxu_data/filtered_isoforms_AA.fasta

