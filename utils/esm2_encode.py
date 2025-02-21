### IMPORTS ###
import sys
from pathlib import Path
import pickle
import time
import torch
import esm
import numpy as np
import logging

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_esm2_encs(data):
    """
    Compute per residue esm2 representation  
    
    input: data: list of tuples: [(seq_name, sequence)...]
    output: sequence_representations: list of esm2 torch tensors, same order data: [esm2_enc1, esm2_enc2...]  
    
    """
    # load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    
    # extract per-residue representations
    with torch.no_grad(): results = model(batch_tokens, repr_layers=[33], need_head_weights=True, return_contacts=True)
    token_representations = results["representations"][33]
    logger.info(f"ESM-2 representation size of sequence batch with padding: {token_representations.size()}")
    
    # omitting embedding from padding, as well as start and end tokens
    sequence_representations = []
    attentions = []
    for i, tokens_len in enumerate(batch_lens):
        batch_token = batch_tokens[i]

        # Extract attention matrices from the final layer
        att = results["attentions"][i][-1]
        # Average the attention weights across heads
        att_cross_head = att.mean(dim=0) 
        
        # if there is cls, eos or padding token, exclude them
        if batch_token[0] == alphabet.cls_idx and (batch_token[-1] == alphabet.eos_idx or batch_token[-1] == alphabet.padding_idx):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1])
            # Average the attention weight for each row afer removing padding, start, and end tokens
            # attentions.append(att_cross_head[1:tokens_len-1, 1:tokens_len-1].mean(dim=0)) 
            attentions.append(att_cross_head[1:tokens_len-1, 1:tokens_len-1]) 
            
        # if not just add as is             
        else: sequence_representations.append(token_representations[i, :, :])

    for s in sequence_representations: 
        logger.info(f"ESM-2 representation size of sequence  after removing padding: {s.size()}") 
    
    gpu_mem = torch.cuda.max_memory_allocated(device=None) / (1024**2)
    torch.cuda.reset_peak_memory_stats(device=None)

    return sequence_representations, attentions, gpu_mem
    

def read_acc_seqs_from_fasta(infile_path, start_offset=-1, batch_size=50):
    """
    Read sequences and accessions from a FASTA file into chunks of desired batch size.
    
    input: infile_path: str, path to the FASTA file
           start_offset: int, offset to start reading from (default: -1, start from the beginning)
           batch_size: int, size of batch to read (default: 50)
    output: accs: list of str, definition lines
           sequences: list of str, sequences
    """
    accs = list()
    sequences = list()

    try:
        seq = ""
        acc = ""
        read_acc = False
        with open(infile_path, "r") as infile:
            n_lines = 0
            offset = -1

            for line in infile:

                offset += 1
                if offset <= start_offset:
                    continue

                line = line.strip()

                if line.startswith(">"):
                    
                    n_lines += 1
                    if n_lines > batch_size:
                        break

                    acc = line.split(">")[1]
                    
                    if read_acc:
                        sequences.append(seq)
                        seq = "" # Reset sequence string
                    accs.append(acc)

                else:
                    seq += line
                    read_acc = True

        if n_lines != 0:
            # Get last sequence
            sequences.append(seq)
    except FileNotFoundError:
        logger.info(f"File not found: {infile_path}")
        return 0
    
    accs_and_sequences = tuple( zip(accs, sequences) )
    return accs_and_sequences, offset-1


def read_acc_seqs_from_fasta_old(infile_path):
    """
    Read sequences and accessions from a FASTA file.
    
    input: infile_path: str, path to the FASTA file
    output: accs: list of str, definition lines
           sequences: list of str, sequences
    """
    accs = list()
    sequences = list()
    seq = ""
    read_acc = False

    infile = Path(infile_path)
    if not infile.is_file():
        logger.info(f"The input file was invalid. Invalid file was {infile}")

    infile = open(infile, "r")
    readfile = infile.readlines()
    infile.close()

    for line in readfile:
        line = line.strip()
        if line.startswith(">"):
            acc = line.split(">")[1]
            if read_acc:
                accs.append(acc)
                sequences.append(seq)
                #reset sequence string
                seq = ""
            #catch first accesion.
            else:
                accs.append(acc)
        else:
            seq += line
            read_acc = True

    #get last sequence
    sequences.append(seq)
    accs_and_sequences = tuple( zip(accs, sequences) )
    return accs_and_sequences
