# Knowledge Discovery Project - Anomaly Detection in Time Series

Knowledge Discovery Project
Data Mining and Time Series – UPM (2025/2026)

Authors: Ottavia Biagi, Leandro Duarte, Emanuele Alberti

Overview

This repository contains the complete Knowledge Discovery Project for the Data Mining and Time Series course at Universidad Politécnica de Madrid.

The project follows the classical KDD (Knowledge Discovery in Databases) process and is structured into two main stages:

Stage 1: Domain Understanding, Data Understanding and Project Goals

Stage 2: Transformer-based Anomaly Detection on real-world benchmarks

Additionally, the repository includes four intermediate assignments covering simulation, clustering, symbolic representation, and domain analysis.

Stage 1 — Domain & Data Understanding

Document: Project_Proposal.pdf


Stage 1 introduces the analytical foundation of the KDD process applied to multivariate time-series anomaly detection.

1. Introduction

The project examines deep learning–based anomaly detection for multivariate time series, with inspiration from state-of-the-art models such as OmniAnomaly and TranAD.
The motivation spans several real-world domains including aerospace telemetry, industrial monitoring and infrastructure logs.

2. Domain Understanding

We analyse three widely used benchmark domains:

NASA SMAP (Soil Moisture Active Passive),

NASA MSL (Mars Science Laboratory),

SMD (Server Machine Dataset) — resource metrics from 28 production servers.

Challenges include:

multivariate dependencies,

rare and heterogeneous anomalies (4–13%),

temporal context modelling,

unsupervised training conditions.

3. Data Understanding

Dataset summary:

Dataset	Entities	Dimensions	Train Samples	Anomaly %
SMAP	55	25	135,183	13.13%
MSL	27	55	58,317	10.72%
SMD	28	38	708,405	4.16%

Preprocessing includes:

Normalisation to [0,1]

Sliding-window extraction (length 10–50)

Handling imbalanced anomaly distributions

Separation by machine groups

4. Project Goals

Stage 1 defines the methodological goals:

Implement a baseline (e.g., LSTM Autoencoder)

Progressively extend to a transformer-based architecture

Analyse reconstruction-based anomaly scoring

Compare performance using Precision, Recall, F1, ROC-AUC

Relate results directly to OmniAnomaly and TranAD

Stage 2 — Transformer-based Anomaly Detection

Full report (Overleaf):
https://www.overleaf.com/9614831219mnzbpmczhwtw#deeb77

Stage 2 consists of the full implementation and evaluation of deep learning models for multivariate anomaly detection using the Server Machine Dataset (SMD).

Main Components
1. Baseline Models

LSTM Autoencoder

Transformer Autoencoder

Training uses windowed input sequences (10–50 timesteps) and min–max normalisation.

2. Reconstruction-based Anomaly Scoring

Anomalies are detected based on the reconstruction error per timestep or per window.

A key empirical issue identified:

~480 windows produce extremely large reconstruction errors (≈ 3.18 × 10¹⁴)

while the rest lie in the range 1e−5–1e−3

This behaviour is analysed as a symptom of:

limited stability of the simplified transformer

lack of stochasticity / variational regularisation

absence of adversarial components (TranAD)

insufficient denoising mechanisms

3. Comparison with Literature

The report discusses differences between our simplified architecture and
state-of-the-art systems such as:

OmniAnomaly (stochastic-VAE + GRU)

TranAD (dual-attention transformer + adversarial training)

These comparisons explain performance gaps and guide future improvements.

Stage 2 Notebook

The notebook containing the full implementation of the anomaly detection experiments is included:

src/stage2_notebook.ipynb


This notebook corresponds directly to the methodology described in the Overleaf report.

Assignments

This repository includes the four official assignments required for the course, stored in the assignments/ folder.

Assignment 1 — Time-Series Simulation & Domain Analysis

Simulated two distinct families of time series (Finance vs Hotel ADR) using the Time Series Random Configurator.
Includes distance-based evaluation and critique of the simulator’s ability to reproduce real structural patterns.


Assignment 2 — Fourier/PAA Clustering with DMonTS

Applied normalisation, dimensionality reduction (Fourier, PAA), and K-Means clustering (K=2,3,4).
Results show perfect domain separation for K=2 and seasonal sub-clusters for Hotel ADR at higher K.


Assignment 3 — Symbolic Representations: SAX & nSDL

Explored symbolic encodings for time series (SAX, nSDL), analysing sensitivity to segments and alphabet size.
Included a Python experiment (bag-of-symbols + K-Means).


Assignment 4 — Domain Interpretation & Structural Analysis

Detailed examination of real-world time-series patterns in Finance and Hotel ADR, comparison with simulator output, and assessment of within/between-domain distances.

## Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd FinalProject

# Sync dependencies with uv
uv sync

# Add a new package (if needed)
uv add <package-name>

# Run Python with uv
uv run python <script.py>

# For Jupyter notebooks
uv run jupyter lab
```

For more details on uv, see the [official documentation](https://docs.astral.sh/uv/).
