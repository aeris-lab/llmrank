# Inter-LLM Divergence – Minimal Reproducibility Notebook

This repository provides a lightweight, fully reproducible implementation of the
core analysis from the paper **“Retrieval Without Consensus: Quantifying Inter-LLM Divergence in API Discovery.”**

## Overview
The notebook `aaai_lamas_study.ipynb` computes four ranking-similarity metrics between large language models (LLMs) across multiple query domains:

| Metric | Description |
|---------|-------------|
| **Average Overlap (AO)** | Fraction of shared items averaged across ranks |
| **Jaccard Similarity** | Intersection / Union of top-K API sets |
| **Rank-Biased Overlap (RBO)** | Weighted overlap emphasizing early ranks |
| **Kendall Tau (τ)** | Rank-order correlation of shared APIs |

The goal is to quantify agreement and divergence between LLMs performing the same retrieval or reasoning task. This version is an optimized version of the study. 

## Requirements
```bash
pip install pandas numpy scipy
```

## Input Format
Provide an Excel file `variant_report_c.xlsx` where **each sheet** corresponds to
a task or query and includes at least these columns:

| LLM Name | Service_Name | Rank |
|-----------|---------------|------|
| ChatGPT   | api.weather   | 1    |
| Claude    | api.weather   | 1    |
| …         | …             | …    |

## Running
To execute locally:
```bash
python aaai_study_minimal.py
```
Or open in Google Colab:
```python
!pip install pandas numpy scipy
!python aaai_study_minimal.py
```

### Output
```
results/pairwise_metrics.csv
```

Example output:
| Query | LLM1 | LLM2 | AO | Jaccard | RBO | KendallTau |
|--------|------|------|----|----------|-----|-------------|
| Weather | ChatGPT | Claude | 0.57 | 0.42 | 0.36 | 0.62 |

## Citation
TBD

## License
MIT License © 2025 
