import os
import sys
import time
import pickle
import logging
import argparse
import datetime
from utils.esm2_encode import  get_esm2_encs, read_acc_seqs_from_fasta 
from utils.aggregate_results import aggregate_tensors
from utils.performance_assess import get_cpu_usage, get_memory_usage


# Configure the logger
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
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


# Set parameters
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--input", help="Input fasta file.")
parser.add_argument("--len_assessment", action='store_true', help="Assess length.")
parser.add_argument("--no_save", action='store_true', help="Don't save results to pickle files.")
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()

DO_ASSESS_LEN = args.len_assessment
SAVE_FILE = not args.no_save
batch_size = args.batch_size


# Check if the input fasta file exists
if not os.path.exists(args.input) and \
    not (args.input.endswith("fasta") or args.input.endswith("fa")):
    logger.error("Invalid input fasta file.")
    exit(1)
file_path = args.input
logger.info('Processing:', file_path)
# Get the directory containing the input fasta file
directory = os.path.dirname(os.path.abspath(file_path))


if DO_ASSESS_LEN:
    logger.info(">>batch_id,length(bp),time(s),cpu(%),gpu_mem,mem(MB)")


# Create the output directories
outname = file_path.rsplit('/',1)[1].rsplit('.')[0]
if not os.path.exists(f"{directory}/{outname}"):
    os.mkdir(f"{directory}/{outname}")
if not os.path.exists(f"{directory}/{outname}/attention"):
    os.mkdir(f"{directory}/{outname}/attention/")
if not os.path.exists(f"{directory}/{outname}/esm2enc"):
    os.mkdir(f"{directory}/{outname}/esm2enc/")


# Read fasta into a list of tuples: (acc1, seq1), (acc2, seq2)...] 
i_batch = 0
offset = -1

while True:
    accs_seqs, offset = read_acc_seqs_from_fasta(
                            file_path, offset, batch_size=batch_size)
    if len(accs_seqs) == 0:
        logger.info("Done!")
        break
    else: 
        logger.info("Batch {}: {} sequences.".format(i_batch, len(accs_seqs)))

        # Filter sequences if the sequence length is longer than 1024
        remove_indices = []
        for i, pair in enumerate(accs_seqs):
            if len(pair[1]) > 1024:
                remove_indices.append(i)
        filtered_list = [item for index, item in enumerate(accs_seqs) 
                         if index not in remove_indices]
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
        output_file = f"{directory}/{outname}/{out_type}.rds"
        aggregate_tensors(input_files, output_file)
        
        # Check if the output file was successfully created
        if os.path.exists(output_file):
            os.system(f"rm -rf {directory}/{outname}/{out_type}/")
            logger.info(f"Original {out_type} files removed.")
        else:
            logger.warning(f"Aggregation failed. Original {out_type} files were not removed.")
