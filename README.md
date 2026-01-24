# Semantic–Structural Enhanced Multi-Order Graph Clustering via Contrastive Learning (S2MoGC)

An official source code for paper **"Semantic–Structural Enhanced Multi-Order Graph Clustering via Contrastive Learning"**.  
Any communications or issues are welcomed. Please contact **[@xxx.com](mailto:your_email@xxx.com)**.

The authors of the paper: 

---

## Overview

We propose **S2MoGC**, a semantic–structural enhanced multi-order graph clustering framework trained with **contrastive learning**.  
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
