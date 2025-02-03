### IMPORTS AND STATIC VARIABLES ###
from pathlib import Path
import os
import sys
import pickle
import torch
from utils.esm2_encode import  get_esm2_encs, read_acc_seqs_from_fasta 
from utils.aggregate_results import aggregate_tensors

if len(sys.argv) < 2:
    print("Usage: python run.py <path_to_your_fasta_file>")
    exit(1)

file_path = sys.argv[1]
print('Processing:', file_path)

# set work directory this file's directory location
directory = os.path.dirname(os.path.abspath(file_path))

### MAIN ###

# read fasta into list tuple format: (acc1, seq1), (acc2, seq2)...] 
if not file_path.endswith("fasta") or not os.path.exists(file_path):
    print("Invalid Data!", file_path)
    exit(1)
else:
    outname = file_path.rsplit('/',1)[1].rsplit('.')[0]
    if not os.path.exists(f"{directory}/{outname}"):
        os.mkdir(f"{directory}/{outname}")
        os.mkdir(f"{directory}/{outname}/attention/")
        os.mkdir(f"{directory}/{outname}/esm2enc/")
    
    i_batch = 0
    offset = -1
    while True:
        accs_seqs, offset = read_acc_seqs_from_fasta(file_path, offset, batch_size=1)
        if len(accs_seqs) == 0:
            print("Done!")
            break
        else: 
            print("Batch {}: {} sequences.".format(i_batch, len(accs_seqs)))

            remove_indices = []
            for i, pair in enumerate(accs_seqs):
                if len(pair[1]) > 1024:
                    remove_indices.append(i)
            filtered_list = [item for index, item in enumerate(accs_seqs) if index not in remove_indices]
            if len(filtered_list) == 0:
                    continue
            esm2_encs, attentions = get_esm2_encs(filtered_list) 
            with open(f"{directory}/{outname}/esm2enc/{outname}_esm2enc_batch{i_batch}.pickle", "wb") as outfile: 
                pickle.dump(esm2_encs, outfile) 
            
            with open(f"{directory}/{outname}/attention/{outname}_attention_batch{i_batch}.pickle", "wb") as outfile: 
                pickle.dump(attentions, outfile) 
            print("-----------------------------", i_batch, "-"*24)
            i_batch += 1
    
    # Aggregate the batches into a single file
    for out_type in ["attention", "esm2enc"]:
        input_files = f"{directory}/{outname}/{out_type}/"
        output_file = f"{directory}/{outname}/{out_type}.pickle"
        aggregate_tensors(input_files, output_file)
        
        # Check if the output file was successfully created
        if os.path.exists(output_file):
            for file in input_files:
                os.remove(file)
            print(f"Original {out_type} files removed.")
        else:
            print(f"Aggregation failed. Original {out_type} files were not removed.")