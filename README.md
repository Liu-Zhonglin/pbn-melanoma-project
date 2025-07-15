# PBN-RL-XAI Pipeline for Discovering Temporal Intervention Strategies in Melanoma Immunotherapy Resistance

This repository contains the complete source code, models, and reproducible analysis pipelines for  
**"A PBN-RL-XAI Framework for Discovering a 'Hit-and-Run' Therapeutic Strategy in Melanoma"**  
([arXiv:2507.10136](https://arxiv.org/abs/2507.10136)), submitted to IEEE BIBM 2025.

---

## Overview

Innate resistance to anti-PD-1 immunotherapy is a major challenge in metastatic melanoma, driven by complex and poorly understood gene regulatory networks. In this project, we:

- **Construct** a dynamic **Probabilistic Boolean Network (PBN)** from patient tumor RNA-seq data ([Hugo et al., Cell 2016](https://pubmed.ncbi.nlm.nih.gov/26997480/)) to model the regulatory logic governing resistance.
- **Discover** optimal, time-dependent therapeutic interventions using **reinforcement learning (RL)** with Proximal Policy Optimization (PPO).
- **Explain** the learned strategies and network mechanisms using **explainable AI (XAI)**, specifically SHAP (SHapley Additive exPlanations).

> **Central finding:**  
> A precisely timed, 4-step *temporary* inhibition of **LOXL2**—rather than sustained inhibition—efficiently erases the molecular signature of resistance. This "hit-and-run" intervention enables the network to self-correct, achieving over **93% reversal** of resistance in silico.

---

## Key Features

- **PBN Model Construction:**  
  - Automated pipeline for binarizing RNA-seq data and inferring context-specific PBNs using a hybrid prior-knowledge/data-driven approach.
  - Core gene selection combines key IPRES genes and data-driven network importance.

- **Network Dynamics & Attractor Analysis:**  
  - Scripts and tools to analyze attractor landscapes and regulatory rewiring that underpins resistance.
  - **Note:** The [optPBN toolbox](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098001) (Matlab) is required for attractor analysis.

- **RL-Based Optimal Control:**  
  - RL training pipeline (using `gym-PBN-stac` and `stable-baselines3`) for discovering policies that transition the network from resistant to sensitive phenotypes.
  - Support for both episode-based and transient ("hit-and-run") interventions.

- **Explainable AI Analysis:**  
  - SHAP-based scripts for interpreting RL agent decisions, including dynamic SHAP trajectory heatmaps and context-dependent vulnerability analysis.

- **Full Reproducibility:**  
  - All data processing, model inference, training, evaluation, and plotting scripts are included and documented.

---

## Folder Structure
- **Data_Preprocessing/**: Input data and scripts for normalization and annotation.  
  *RNA-seq data can be downloaded from GEO accession [GSE78220](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE78220).*
- **PKN&Binarization/**: Prior knowledge network (PKN) assembly and gene binarization scripts.  
  The PKN is derived from the KEGG pathway **PD-L1 expression and PD-1 checkpoint pathway in cancer - Homo sapiens (human)** ([hsa05235](https://www.genome.jp/dbget-bin/www_bget?pathway:hsa05235)).
- **PBN Construction/**: Data-driven PBN inference and network logic selection.
- **Mechanistic_Analysis/**: Attractor and rewiring analysis, phenotype decoding, and results.  
- **Optimal_Control/**: RL training, SHAP/XAI analysis, robust intervention studies.


---



## Environment and Dependencies

- **Python 3.9+**
- **Matlab** (for optPBN attractor analysis)
- **R** (for some preprocessing and network inference scripts)


