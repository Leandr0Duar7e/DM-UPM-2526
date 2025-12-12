Knowledge Discovery Project – Anomaly Detection in Time Series

Data Mining and Time Series — UPM (2025/2026)

Authors: Ottavia Biagi, Leandro Duarte, Emanuele Alberti

Overview

This repository contains the complete Knowledge Discovery Project for the Data Mining and Time Series course at Universidad Politécnica de Madrid.
The project follows the classical KDD (Knowledge Discovery in Databases) methodology and is structured into two main stages:

Stage 1: Domain Understanding, Data Understanding, and Project Goals

Stage 2: Transformer-based Anomaly Detection on real-world benchmarks

The repository also includes five assignments covering simulation, clustering, symbolic representation, ARIMA modelling, and domain analysis.

Stage 1 — Domain & Data Understanding

Document: Project_Proposal.pdf


Stage 1 introduces the theoretical and methodological foundations of the KDD pipeline as applied to multivariate time-series anomaly detection.

1. Introduction

The project examines deep-learning–based anomaly detection strategies and draws inspiration from prominent models including OmniAnomaly and TranAD.
Applications include aerospace telemetry, industrial monitoring, and server infrastructure logs.

2. Domain Understanding

Three widely used benchmark datasets are analysed:

NASA SMAP (Soil Moisture Active Passive)

NASA MSL (Mars Science Laboratory)

Server Machine Dataset (SMD)

Key challenges include multivariate dependencies, rare anomalies (4–13%), non-stationary patterns, temporal structure, and unsupervised learning constraints.

3. Data Understanding
Dataset	Entities	Dimensions	Train Samples	Anomaly %
SMAP	55	25	135,183	13.13%
MSL	27	55	58,317	10.72%
SMD	28	38	708,405	4.16%

Preprocessing steps include:

Min–max normalisation

Sliding-window segmentation (length 10–50)

Handling imbalanced anomaly distributions

Per-machine separation

4. Project Goals

Stage 1 defines the main objectives:

Implement a baseline LSTM autoencoder

Develop a simplified transformer autoencoder

Perform reconstruction-based anomaly scoring

Evaluate models using Precision, Recall, F1, ROC-AUC, PR-AUC

Relate empirical results to OmniAnomaly and TranAD

Stage 2 — Transformer-based Anomaly Detection

Stage 2 report (Overleaf):
https://www.overleaf.com/9614831219mnzbpmczhwtw#deeb77

Stage 2 report PDF: DM_secondStage.pdf


Stage 2 implements the modelling, training, and evaluation steps of the KDD process on machine-1-1 of the Server Machine Dataset (SMD). The analysis includes baseline models, transformer variants, thresholding strategies, error distribution studies, and feature-level diagnostics.

Models Implemented

Mean reconstruction baseline

LSTM Autoencoder

Transformer Autoencoder (simplified)

Ablation variants:

LayerNorm

40-epoch training

Huber loss

High dropout

Mixed positional encoding

Output normalisation

Main Results (Timestamp-Level Detection)

Results (from Stage 2 PDF, pages 8–12):

Model	Precision	Recall	F1	ROC-AUC	PR-AUC
Mean baseline	0.57	0.47	0.52	0.91	0.58
LSTM AE	0.19	0.99	0.31	0.88	0.52
Transformer AE (LN, 20 epochs)	0.34	0.45	0.39	0.87	0.44
Transformer AE (LN, 40 epochs)	0.37	0.55	0.44	0.89	0.47

The LayerNorm transformer trained for 40 epochs is the best-performing learned model in this project.

Reconstruction Error Phenomenon

Approximately 480 windows produced extremely large reconstruction errors (up to around 3.18 × 10¹⁴).
After filtering non-finite and implausible values, the remaining error range stabilised between 9.4 × 10⁻⁵ and 1.06.

This behaviour is attributed to the simplicity of the model and the absence of stabilising components such as adversarial refinement, deeper attention blocks, or self-conditioning.

Extended Analyses (Stage 2)

The final report includes:

Error distribution analysis (normal vs anomalous windows)

Zoomed reconstruction–vs–raw-signal comparison

Threshold sensitivity analysis

Feature-level anomaly diagnosis (per-feature MSE heatmaps)

Comparison with TranAD and OmniAnomaly

Architectural differences table summarising key methodological gaps

Stage 2 Notebook

All modelling and experimental procedures are implemented in:

src/stage2_SMD_time_series_anomaly_detection.ipynb

Assignments (1–5)

All assignments are stored in the assignments/ directory.

Assignment 1 — Time-Series Simulation & Domain Analysis

Simulation of two domains (Finance vs Hotel ADR) using the Time Series Random Configurator, followed by distance-based evaluation.


Assignment 2 — Fourier/PAA Clustering with DMonTS

Dimensionality reduction (Fourier, PAA) followed by K-Means clustering for K=2–4.


Assignment 3 — Symbolic Representations (SAX & nSDL)

Study of symbolic encodings and their impact on clustering performance. Includes Python implementation with bag-of-symbols and K-Means.


Assignment 4 — Advanced Domain Interpretation

Analysis of structural patterns for Finance and Hotel ADR. Evaluation of the simulator’s ability to reproduce real-world behaviours.


Assignment 5 — ARIMA Modelling and Forecasting

Complete Box–Jenkins workflow applied to the Student time series, including EDA, ACF/PACF analysis, ARIMA model fitting, forecasting, and residual diagnostics.
Final model: ARIMA(1,0,1).


Quick Setup
# Clone the repository
git clone <repository-url>
cd DM-UPM-2526

# Sync dependencies
uv sync

# Add a new package
uv add <package-name>

# Run Python scripts
uv run python <script.py>


# Launch Jupyter Lab
uv run jupyter lab
