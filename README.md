# saccadic_based-classification-using-nested-vallidation
Saccadic eye movement–based classification of OCD, schizophrenia, and healthy controls using a shallow neural network with nested cross-validation. MATLAB implementation supporting reproducible analysis of oculomotor biomarkers.

Overview

This project implements a shallow neural network for classifying saccadic eye movement data using nested cross-validation.

If you use this code, please cite:

Koliaraki, M.-N., Smyrnis, N., Asvestas, P., Matsopoulos, G. K., & Ventouras, E.-C. (2026).
Saccadic eye movements based classification of patients with obsessive-compulsive disorder, patients with schizophrenia and healthy controls using artificial neural networks.
Cognitive Neurodynamics, 20(1), 41. Springer Netherlands.

Description

This repository contains the MATLAB implementation of the neural network classification framework described in the above paper, including nested cross-validation and feature-based classification of oculomotor behavior

Method
External: 3-Fold Cross-Validation
Internal: Leave-One-Out (LOO)
Model: MATLAB Neural Network (patternnet)


How to Run
Load:
task2g12_inputs []
task2g12_targets

Run the script:
saccadic_classification_nested_validation

Notes
Make sure Neural Network Toolbox is installed
Data is not included due to privacy 


Author

M.-N. Koliaraki (2025)
