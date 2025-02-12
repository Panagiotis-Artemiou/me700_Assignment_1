# ME700 Assignment 1
This repository includes code afor Assigment 1 for the ME700 class at BU, spring semester 2025.

## Conda environment, install, and testing <a name="install"></a>
First instal miniconda
```bash
module load miniconda
```

To install this package, please begin by setting up a conda environment (mamba also works):
```bash
conda create --name newtons-method-env python=3.12
```
Once the environment has been created, activate it:

```bash
conda activate newtons-method-env
```
Double check that python is version 3.12 in the environment:
```bash
python --version
```
Ensure that pip is using the most up to date version of setuptools:
```bash
pip install --upgrade pip setuptools wheel
```
Create an editable install of the bisection method code (note: you must be in the correct directory):
```bash
pip install -e .
```
Test that the code is working with pytest:
```bash
pytest -v --cov=noewtonsmethod --cov-report term-missing
```
Code coverage should be 100%.
