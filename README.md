## Introduction
This repository contains the implementation of extracting encodings and attention layers from esm2 models.
The implementation is based on [esm2_uilities from mnielLab](https://github.com/mnielLab/esm2_utilities).

## Installation
* Set up a conda environment by using environment.yml and activate the environment.
```bash
conda env create -f environment.yml -n esm2
```
* Clone this repository.
```bash
git clone https://github.com/chihs-dtu/esm-utils.git
```
* Test the environment with the example data.
```bash
bash test.sh
```