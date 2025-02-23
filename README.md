## Introduction
This repository contains the implementation of extracting encodings and attention layers from esm2 models.
The implementation is based on [esm2_uilities from mnielLab](https://github.com/mnielLab/esm2_utilities).

## Installation
* Set up a conda environment by using environment.yml and activate the environment.
```bash
conda env create -f environment.yml -n esm2
conda activate esm2
```
* Clone this repository.
```bash
git clone https://github.com/chihs-dtu/esm-utils.git
```
* Test the environment with the example data.
```bash
bash test.sh
```
* Run with your data.
```bash
python run.py --input $PATH_TO_YOUR_DATA(.fasta)
```
## Load JSON file into R
```R
library(jsonlite)

attn <- fromJSON('attention.json')
enc <- fromJSON('esm2enc.json')
```
