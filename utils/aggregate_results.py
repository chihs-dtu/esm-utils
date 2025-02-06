import os,sys
import pickle
import logging
import re
import json

logger = logging.getLogger(__name__)

def extract_batch_number(filename):
    """
    Extracts the batch number from a filename, assuming it follows the pattern 'batch<number>.pickle'.
    Returns infinity if no batch number is found.
    
    Args:
    filename (str): The filename to extract the batch number from.
    
    Returns:
    int: The batch number extracted from the filename, or infinity if no batch number is found.
    """
    # Use regular expression to extract the batch number (assumed to be after 'batch' and followed by digits)
    match = re.search(r'batch(\d+)', filename)
    if match:
        return int(match.group(1))  # Return the batch number as an integer
    else:
        return float('inf')  # If no batch number is found, place it at the end (or skip)


def aggregate_tensors(pickle_dir, output_file):
    """
    Aggregates all tensors from pickle files in a given directory, sorting them by batch number.
    Stores the aggregated tensors in a new pickle file.
    
    Args:
    pickle_dir (str): The directory containing the pickle files.
    output_file (str): The path to the output pickle file.
    """
    if output_file.endswith('.json'):
        save_json = True
    else:
        save_json = False

    # This will hold all the tensors in the order they appear
    all_tensors = []
    
    # Get a list of all pickle files in the directory
    pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith(".pickle")]
    
    # Sort the files by batch number extracted from the filename
    pickle_files.sort(key=extract_batch_number)
    
    # Loop over all the sorted pickle files
    for filename in pickle_files:
        filepath = os.path.join(pickle_dir, filename)
        with open(filepath, 'rb') as file:
            # Load the list of tensors from the pickle file
            tensor_list = pickle.load(file)
            
            # Ensure it's a list of tensors
            if isinstance(tensor_list, list):
                if save_json:
                    all_tensors.extend([x.tolist() for x in tensor_list])
                else:
                    all_tensors.extend(tensor_list)
            else:
                logger.info(f"Warning: Skipping file {filename} because it doesn't contain a list.")
    
    if save_json:
        # Save the aggregated tensors to the output json file
        with open(output_file, 'w') as file:
            json.dump(all_tensors, file)
    else:
        # Save the aggregated tensors to the output pickle file
        with open(output_file, 'wb') as file:
            pickle.dump(all_tensors, file)
    
    logger.info(f"Aggregated {len(all_tensors)} tensors to {output_file}")