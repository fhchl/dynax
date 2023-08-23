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
    affiliation: 1
  - name: Finn T. Agerkvist
    orcid: 0000-0001-9434-9008
    affiliation: 1
affiliations:
 - name: Department of Electrical and Photonical Engineering, Technical University of Denmark, Denmark
   index: 1
date: XX August 2023
bibliography: paper.bib
---

# Summary

`Dynax` is a Python package for modeling nonlinear dynamical systems,
identifying their parameters from data, and designing linearizing feedback laws
for controlling their input-output behavior. Its main features include:

- estimation of ODE parameters and their covariance via the prediction-error method
- estimation of linear ODE parameters via matching of transfer-functions
- fitting of multiple experiments
- fitting with multiple shooting
- input-output linearization of continuous-time input-affine systems with well-defined relative degree
- input-output linearization of discrete-time systems with well-defined relative degree
- estimation of a system's relative-degree

# Statement of need

- nonlinear compensation: no open-source software at all

Currently, there exists no tools in the Python ecosystem that directly facilitate parameter estimation for nonlinear differential equation systems (sometimes called "grey-box models"). There exists
[`nlgreyest`](https://se.mathworks.com/help/ident/ref/nlgreyest.html) in Matlab's Control
System toolbox, [`DiffEqParamEstim.jl`](https://docs.sciml.ai/DiffEqParamEstim/stable/) in Julia, [`sysidentpy`](https://github.com/wilsonrljr/sysidentpy)[@lacerda2020sysidentpy] and [`SIPPY`](https://github.com/CPCLAB-UNIPI/SIPPY) in Python.

`Dynax` was designed to be used by researchers, students and engineers. It
is already used in a course at DTU for estimating the parameters of loudspeaker drivers instead of a commercial black-box solution.
The combination of design and ease of use make it possible to iteratively test and
develop dynamical system models for linearization of such transducers.

# Similar software

- Dynamax
- nlgreyest
- nlgreyfast
- DiffEqParamEstim

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References

