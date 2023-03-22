# Cosmology-informed neural networks

# Introduction

This repository contains Python scripts related to the following work: 
[Cosmology-informed neural networks to solve the background dynamics of the Universe](https://arxiv.org/abs/2205.02945) (DOI: [10.1103/PhysRevD.107.063523](https://doi.org/10.1103/PhysRevD.107.063523)).
The aim of this repository is to share examples of how to train neural netowrks to be bundle solutions of a differential system.
In particular, the examples here concern the background dynamics of the Universe for four different cosmological models:
* The Lambda Cold Dark Matter model $\left(\Lambda \mathrm{CDM}\right)$.
* The parametric dark energy model Chevallier-Polarski-Linder (CPL).
* A quintessence model with an exponential potenial.
* The cosmology under the $f\left(R\right)$ Hu-Sawicki gravity theory.

The scripts are written using the [PyTorch](https://github.com/pytorch/pytorch) based [neurodiffeq](https://github.com/NeuroDiffGym/neurodiffeq) library.

# Usage
## Set Up
To run these scripts one can clone this repository, set up a virtual enviroment, and install the necessary dependencies in it:

```
git clone https://github.com/at-chantada/cosmo-nets
conda create -n cosmonet_venv
source activate cosmonet_venv
pip install -r requirements.txt

```
## Runing the scripts
For each model there is a script that trains the netowrk, and another that loads it to make use of it. The former trains a neural network to solve 
the model's differential equation, and saves it in the current directory along with a plot of the total loss during training. 
Then, the corresponding loading scirpt can be used to load the saved network and use it. In the example it is used to plot the Hubble parameter for different
values of the model's parameters, and save that plot.

Keep in mind that to run any script in this repository (exept the ones that correspond to $\Lambda \mathrm{CDM}$), the script must be located in the same directory as where the `utils.py` script (from this repository) is located.

# API Reference
Because the scripts use the [neurodiffeq](https://github.com/NeuroDiffGym/neurodiffeq) library, there are classes such as `BundleSolver1D` and `Generator1D`.
The detailed descriptions of these can be found in [neurodiffeq's API Reference](https://neurodiffeq.readthedocs.io/en/latest/api.html). 
