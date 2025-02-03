import time
import psutil
import subprocess
import os
import sys

def get_gpu_usage():
    # Run 'nvidia-smi' to get GPU stats (only once)
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    
    # Parse the output to include units
    gpu_usage = int(result.stdout.strip())

    return gpu_usage

def get_cpu_usage():
    # Get the current CPU usage (percentage)
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    # Get memory usage (in MB)
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 ** 2  # Memory usage in MB

def main():
    # Track the start time
    start_time = time.time()

    # Example code block (simulate some work)
    print("Starting the process...")
    result = subprocess.run(
        ['/home/people/chihs/miniconda3/envs/esm2/bin/python', 'run.py', sys.argv[1]],
        stdout=None
    )

    # Track the end time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Get CPU, GPU, and memory usage
    cpu_usage = get_cpu_usage()
    gpu_usage = get_gpu_usage()
    memory_usage = get_memory_usage()

    # Print results
    print("\nPerformance Stats:")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"CPU Usage: {cpu_usage}%")
    print(f"GPU Memory Usage: {gpu_usage}")
    print(f"Memory Usage: {memory_usage:.2f} MB")

if __name__ == "__main__":
    main()
