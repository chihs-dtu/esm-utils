### IMPORTS AND STATIC VARIABLES ###
from pathlib import Path
import os
import sys
import datetime
import time
import logging
import pickle
import torch
from utils.esm2_encode import  get_esm2_encs, read_acc_seqs_from_fasta 
from utils.aggregate_results import aggregate_tensors
from utils.performance_assess import get_cpu_usage, get_gpu_usage, get_memory_usage


# Set parameters
DO_ASSESS_LEN = True
SAVE_FILE = True
batch_size = 1 if DO_ASSESS_LEN else 10


if len(sys.argv) < 2:
    print("Usage: python run.py <path_to_your_fasta_file>")
    exit(1)
file_path = sys.argv[1]
print('Processing:', file_path)
# find the work directory this file's directory location
directory = os.path.dirname(os.path.abspath(file_path))
# Get current date in the desired format
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log format
    datefmt="%Y/%m/%d %H:%M:%S",  # Customize the date format
    handlers=[
        logging.FileHandler(f"run_{current_date}.log", mode='w'),
        logging.StreamHandler()]
)
# Create a logger instance
logger = logging.getLogger(__name__)
if DO_ASSESS_LEN:
    logger.info(">>batch_id,length(bp),time(s),cpu(%),gpu_mem,mem(MB)")


# read fasta into list tuple format: (acc1, seq1), (acc2, seq2)...] 
if not file_path.endswith("fasta") or not os.path.exists(file_path):
    logger.error("Invalid Data!", file_path)
    exit(1)
else:
    outname = file_path.rsplit('/',1)[1].rsplit('.')[0]
    if not os.path.exists(f"{directory}/{outname}"):
        os.mkdir(f"{directory}/{outname}")
    if not os.path.exists(f"{directory}/{outname}/attention"):
        os.mkdir(f"{directory}/{outname}/attention/")
    if not os.path.exists(f"{directory}/{outname}/esm2enc"):
        os.mkdir(f"{directory}/{outname}/esm2enc/")
    
    i_batch = 0
    offset = -1

    while True:
        accs_seqs, offset = read_acc_seqs_from_fasta(file_path, offset, batch_size=batch_size)
        if len(accs_seqs) == 0:
            logger.info("Done!")
            break
        else: 
            logger.info("Batch {}: {} sequences.".format(i_batch, len(accs_seqs)))

            # Filter sequences if length is more than 1024
            remove_indices = []
            for i, pair in enumerate(accs_seqs):
                if len(pair[1]) > 1024:
                    remove_indices.append(i)
            filtered_list = [item for index, item in enumerate(accs_seqs) if index not in remove_indices]
            if len(filtered_list) == 0:
                    continue
            
            # Track the start time and cpu usage
            start_time = time.time()
            cpu_mem_start = get_memory_usage()

            # Run the process and record the gpu mem usage
            esm2_encs, attentions, gpu_mem = get_esm2_encs(filtered_list) 

            # Track the end time
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if DO_ASSESS_LEN:
                # Get CPU, GPU, and memory usage
                cpu_usage = get_cpu_usage()
                cpu_mem_end = get_memory_usage()

                logger.info(">>batch_{},{},{},{},{},{:.2f}".format(
                                                    i_batch,
                                                    len(accs_seqs[0][1]),
                                                    elapsed_time, 
                                                    cpu_usage, 
                                                    gpu_mem,
                                                    cpu_mem_end - cpu_mem_start))
            if SAVE_FILE:
                with open(f"{directory}/{outname}/esm2enc/{outname}_esm2enc_batch{i_batch}.pickle", "wb") as outfile: 
                    pickle.dump(esm2_encs, outfile) 
                
                with open(f"{directory}/{outname}/attention/{outname}_attention_batch{i_batch}.pickle", "wb") as outfile: 
                    pickle.dump(attentions, outfile) 

            i_batch += 1
    
    if SAVE_FILE:
        # Aggregate the batches into a single file
        for out_type in ["attention", "esm2enc"]:
            input_files = f"{directory}/{outname}/{out_type}/"
            output_file = f"{directory}/{outname}/{out_type}.pickle"
            aggregate_tensors(input_files, output_file)
            
            # Check if the output file was successfully created
            if os.path.exists(output_file):
                os.system(f"rm -rf {directory}/{outname}/{out_type}/")
                logger.info(f"Original {out_type} files removed.")
            else:
                logger.warning(f"Aggregation failed. Original {out_type} files were not removed.")
