---
title: 'Dynax: A Python package for parameter estimation and linearization of dynamical system'
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
affiliations:
 - name: Department of Electrical and Photonical Engineering, Technical University of Denmark, Denmark
   index: 1
date: XX August 2023
bibliography: paper.bib
---

# Summary

Describing the evolution of systems over time is an integral part of the sciences and engineering for verifying models,
predicting outcomes, and controlling systems for some desired behavior. However,
the parameters of such models are often unknown and need to be estimated from
data. `Dynax` is a Python package for modeling nonlinear dynamical
systems and estimating their parameters from data. Additionally, it provides
routines for computing feedback laws which render the input-output behavior of
these systems linear and thus controllable. `Dynax` is based on JAX and uses its automatic differentiation for speeding up optimizations and computing the input-output linearizing control inputs automatically from model descriptions.

`Dynax`'s main features include:

- Parameters estimation of nonlinear systems: fitting of both continuous and discrete-time systems to data via the prediction-error method [@ljung2002prediction], fitting to multiple experiments simultaneously, fitting via multiple shooting [@Bock1981;@9651533], estimation of parameter covariances, and box-constraints on parameter values.
- Automatic input-output linearization [@sastryNonlinearSystems1999]: computing of feedback laws for both continuous-time input-affine systems and general discrete-time systems with well-defined relative degrees that allow tracking of linear reference outputs.
- Parameter estimation of linear or linearized ODEs via matching of frequency-responses [@pintelonSystemIdentificationFrequency2012]: this is helpful for obtaining good starting guesses for the identification of the nonlinear identification.


# Statement of need


Currently, there exist no tools in the Python ecosystem that directly facilitate parameter estimation for nonlinear differential equation systems (sometimes called "grey-box models"). For nonlinear system identification, there exists `nlgreyest`[^nlgreyest] and `nlgreyfast` [@retzler_shooting_2022] in Matlab, `SciMLSensitivity` [@rackauckas2020universal] and the related SciML ecosystem in Julia. For Python, there exists only packages for linear system identification like `SIPPY`^[https://github.com/CPCLAB-UNIPI/SIPPY] or non-parameteric models like `sysidentpy` [@lacerda2020sysidentpy]. Most importantly, Dynax seems to be the first publicly available software package for computing input-output linearizing control signals automatically.

`Dynax` was designed to be used by researchers, students and engineers that are familiar with Python without the need to know about optimization, estimation and automatic differentiation. Importantly, only minimal use of JAX is required by the user when defining dynamical systems. `Dynax` is already used at the Technical University of Denmark for research and teaching on the modeling of nonlinear acoustic transducers. There, it is enjoyed for the simplicity with which one can come up with new models, fit them to data and automatically compute the signals that make such transducers act more linearly and thus less distorted.

[^nlgreyest]: https://se.mathworks.com/help/ident/ref/nlgreyest.html


# Acknowledgements

We acknowledge discussions with from Manuel Hahmann and Finn T. Agerkvist that helped shape this library. This research was supported by a research collaboration with Huawei.

# References

