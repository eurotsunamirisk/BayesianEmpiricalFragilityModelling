# Data-Driven Bayesian Empirical Fragility Modelling

**Author(s):** Hossein Ebrahimian, Jingren Wu, Fatemeh Jalayer

## Description
This Jupyter Notebook presents the workflow for data-driven Bayesian empirical fragility modelling. It illustrates how the fragility data provided in the EPOS-ICS-C portal under the layer “Empirical Tsunami Risk Products Dataset (ETRiS v0)” can be generated.
The workflow uses observed damage and flow depth data from past tsunami events. The Notebook derives empirical fragility curves and their confidence bands through Bayesian inference using MCMC simulation. It applies a generalized Binomial regression model with three possible link functions: logit, probit, and cloglog. The approach supports hierarchical fragility modelling for a set of mutually exclusive and collectively exhaustive damage states and for different classes of buildings or infrastructure. 
In the current version of the Notebook, empirical fragility is computed for a single building class.
The observed data include information on tsunami effects (e.g., tsunami height, flow depth) and tsunami consequences (e.g., casualties, building damage). The resulting fragility curves express the probability of exceeding a given damage state as a function of a tsunami intensity measure (e.g., flow depth) for a specific building class or, more generally, a particular asset at risk.


## Contents
- `BayesianEmpiricalFragility.ipynb` – main workflow
- `functionlib.py` – modelling functions
- `data/Samoan Building Class 1.csv` – sample dataset

## How to run
1. Install:
   `pip install -r requirements.txt`
2. Start Jupyter:
   `jupyter lab`
3. Open the notebook and run all cells top-to-bottom.

## Citation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18551506.svg)](https://doi.org/10.5281/zenodo.18551506)





