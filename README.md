# neuromorphic-ms
A biologically inspired framework for mass spectral similarity using Spike-Timing-Dependent Plasticity (STDP) and fragmentation hierarchies


# Rethinking Spectral Similarity in Mass Spectrometry: A Neuromorphic Perspective

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19205996.svg)](https://doi.org/10.5281/zenodo.19205996)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code, data, and benchmarking results for the paper **"Rethinking Spectral Similarity in Mass Spectrometry: A Neuromorphic Perspective"** published in *Frontiers in Analytical Science*.

## Overview

Traditional spectral similarity metrics (e.g., Cosine, Jaccard) treat mass spectra as static vectors. This project introduces a **Neuromorphic Similarity Score ($S_{neu}$)** that reframes fragmentation as a temporal sequence. By mapping peak intensity ranks to "neural spikes" and applying Spike-Timing-Dependent Plasticity (STDP) principles, we capture the hierarchical nature of fragmentation cascades.

## Repository Contents

- `/code`: Python implementation of the `neuromorphic_algorithm_v7` and the `Spectralogic AI v10.3` benchmarking suite.
- `/data`: 
    - `query_spectra.msp`: 1,972 diverse query spectra curated from public NIST and MoNA-type sources.
    - `reference_library.msp`: The reference spectral library used for matching.
- `/results`: Raw output files, CSV summaries, and metadata used to generate Figure 1 and the performance metrics (F1-score, Recall) reported in the paper.
- `/docs`: Supplementary documentation regarding the STDP parameter tuning ($\alpha, \beta$).

## Installation

```bash
git clone [https://github.com/SpectralogicAI/neuromorphic-ms.git](https://github.com/SpectralogicAI/neuromorphic-ms.git)
cd neuromorphic-ms
pip install -r requirements.txt
