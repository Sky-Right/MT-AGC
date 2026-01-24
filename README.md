# Multi-Order Feature Fusion and Pseudo-Label Guided Semantic Matching for Contrastive Graph Clustering

An official source code for paper **"Multi-Order Feature Fusion and Pseudo-Label Guided Semantic Matching for Contrastive Graph Clustering"**.  
Any communications or issues are welcomed. Please contact **[@xxx.com](mailto:your_email@xxx.com)**.

The authors of the paper: 

---

## Overview

We propose **MPCGC**, a Multi-Order Feature Fusion and Pseudo-Label Guided Semantic Matching for Contrastive Graph Clustering.  
The framework mainly contains:

- **Multi-order feature smoothing / fusion** on the original graph
    
- **Top-K global semantic enhancement** via KNN-based edges
    
- **Contrastive learning** with Gaussian perturbation
    
- **Pseudo-label based semantic label matching**
---

## Start
- Step1: unzip the dataset into the **./dataset** folder
- Step2: run

```
python train.py --dataset cora
python train.py --dataset citeseer
```
