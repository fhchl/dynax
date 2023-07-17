---
title: 'Dynax: A package for parameter estimation and linearization of dynamical system'
tags:
  - Python
  - control
  - dynamics
  - system identification
  - parameter estimation
  - linearization
authors:
  - name: Franz M. Heuchel
    orcid: 0000-0002-6084-0170
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Finn T. Agerkvist
    affiliation: 2
affiliations:
 - name: Department of Electrical and Photonical Engineering, Technical University of Denmark, Denmark
   index: 1
date: XX August 2023
bibliography: paper.bib

---

# Summary

`Dynax` is a Python package for modeling nonlinear dynamical systems,
identifying their parameters from data, and designing linearizing feedback laws
for controlling their input-output behaviour.

Features

- Multiple shooting estimation
- Multiple experiment estimation
- 

# Statement of need

- nonlinear compensation: no open-source software at all

Currently, there exists no tools in the Python ecosystem that facilitate parameter estimation 
for differential equation systems (sometimes called "grey-box models"). There exists 
[`nlgreyest`](https://se.mathworks.com/help/ident/ref/nlgreyest.html) in Matlab's Control 
System toolbox and [`DiffEqParamEstim.jl`](https://docs.sciml.ai/DiffEqParamEstim/stable/) in Julia.

`Dynax` was designed to be used by researchers, students and engineers. It
is already used in a course at DTU for estimating the parameters of loudspeaker drivers instead of a commercial black-box solution.
The combination of design and ease of use make it possible to iteratively test and
develop dynamical system models for linearization of such transduers.

# Similar software

- Dynamax
- nlgreyest
- nlgreyfast
- DiffEqParamEstim

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References

