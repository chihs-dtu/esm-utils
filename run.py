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

DO_ASSESS_LEN = True

if len(sys.argv) < 2:
    print("Usage: python run.py <path_to_your_fasta_file>")
    exit(1)

file_path = sys.argv[1]
print('Processing:', file_path)

# set work directory this file's directory location
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
logger.inf(">>length(bp),time(s),cpu(%),gpu_mem,mem(MB)")


# read fasta into list tuple format: (acc1, seq1), (acc2, seq2)...] 
if not file_path.endswith("fasta") or not os.path.exists(file_path):
    logger.error("Invalid Data!", file_path)
    exit(1)
else:
    outname = file_path.rsplit('/',1)[1].rsplit('.')[0]
    if not os.path.exists(f"{directory}/{outname}"):
        os.mkdir(f"{directory}/{outname}")
        os.mkdir(f"{directory}/{outname}/attention/")
        os.mkdir(f"{directory}/{outname}/esm2enc/")
    
    i_batch = 0
    offset = -1
    batch_size = 1 if DO_ASSESS_LEN else 10

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
            
            # Track the start time and gpu usage
            start_time = time.time()
            gpu_usage_start = get_gpu_usage()

            # Run the process
            esm2_encs, attentions = get_esm2_encs(filtered_list) 
            torch.cuda.synchronize()

            # Track the end time
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if DO_ASSESS_LEN:
                # Get CPU, GPU, and memory usage
                cpu_usage = get_cpu_usage()
                gpu_usage_end = get_gpu_usage()
                memory_usage = get_memory_usage()
                logger.info(">>{:.2f},{},{},{},{:.2f}".format(len(accs_seqs[0][1]),
                                                    elapsed_time, 
                                                    cpu_usage, 
                                                    gpu_usage_end - gpu_usage_start,
                                                    memory_usage))
            else:
                with open(f"{directory}/{outname}/esm2enc/{outname}_esm2enc_batch{i_batch}.pickle", "wb") as outfile: 
                    pickle.dump(esm2_encs, outfile) 
                
                with open(f"{directory}/{outname}/attention/{outname}_attention_batch{i_batch}.pickle", "wb") as outfile: 
                    pickle.dump(attentions, outfile) 
            print("-----------------------------", i_batch, "-"*24)
            i_batch += 1
    
    if not DO_ASSESS_LEN:
        # Aggregate the batches into a single file
        for out_type in ["attention", "esm2enc"]:
            input_files = f"{directory}/{outname}/{out_type}/"
            output_file = f"{directory}/{outname}/{out_type}.pickle"
            aggregate_tensors(input_files, output_file)
            
            # Check if the output file was successfully created
            if os.path.exists(output_file):
                for file in input_files:
                    os.remove(file)
                logger.info(f"Original {out_type} files removed.")
            else:
                logger.warning(f"Aggregation failed. Original {out_type} files were not removed.")