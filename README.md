# CNN-GP: Alpha Factor Mining via Representation Learning and Genetic Programming

This repository provides the source code, datasets, and supplementary material for **CNN-GP**, a coarse-to-fine framework for alpha factor mining.

CNN-GP integrates **representation learning** and **symbolic mining** into a unified framework. A Convolutional Neural Network (CNN) is first used to transform the original high-dimensional stock features into a compact latent representation. Genetic Programming (GP) then searches over the learned latent features to generate explicit symbolic alpha factors.

The experiments are conducted on two real-world stock market datasets, **CSI500** and **CSI1000**.
<p align="center">
  <img src="figures/cnn-gp流程图.svg" width="900">
</p>
## Repository Structure

```text
CNN-GP/
│
├── #430-CNN-GP Supplementary Material.pdf
├── GP-CNN.py
├── SCI500_code.xlsx
├── SCI1000_code.xlsx
├── factor_eval.py
└── README.md
```

### File Description

- **`GP-CNN.py`**  
  Main implementation of the CNN-GP framework, including CNN-based latent feature extraction and GP-based symbolic alpha factor mining.

- **`SCI500_code.xlsx`**  
  Data associated with the CSI500 experiments.

- **`SCI1000_code.xlsx`**  
  Data associated with the CSI1000 experiments.

- **`factor_eval.py`**  
  Evaluation code for the generated alpha factors, including the calculation of the main evaluation metrics used in the paper.

- **`#430-CNN-GP Supplementary Material.pdf`**  
  Supplementary material containing additional implementation details, experimental results, statistical significance analysis, runtime analysis, CNN architecture, and parameter sensitivity analysis.

## Method Overview

CNN-GP follows a coarse-to-fine alpha factor mining paradigm:

1. **Representation Learning**

   The CNN learns compact latent representations from the original stock features. Instead of directly performing symbolic search over the high-dimensional raw feature space, CNN-GP first transforms the raw features into a lower-dimensional and task-oriented latent space.

2. **Symbolic Alpha Factor Mining**

   GP subsequently searches over the learned latent features and constructs explicit symbolic expressions as predictive alpha factors.

3. **Factor Evaluation**

   The generated alpha factors are evaluated on the CSI500 and CSI1000 datasets using the evaluation protocol described in the paper and supplementary material.

## Evaluation Metrics

The main evaluation metrics are **RankIC** and **ALER**.

### RankIC

For each trading day, RankIC is calculated as the cross-sectional Spearman correlation between the predicted factor values of all evaluated stocks and their one-day forward returns. The daily RankIC values are then averaged over the entire test period.

### ALER

For each trading day, all evaluated stocks are ranked according to their predicted factor values. The top 10% of stocks are equally weighted to construct the long portfolio.

The daily excess return is calculated as the return of the long portfolio minus the equal-weighted return of all evaluated stocks on the same trading day. The daily excess returns are then compounded over the entire test period and annualized to obtain ALER.

## CNN Architecture

The CNN component performs **feature-oriented representation learning** rather than explicitly modeling dependencies across consecutive trading dates.

The network contains four Conv1D blocks with the following channel configurations:

```text
1 → 64 → 128 → 128 → 256
```

All convolutional layers use a kernel size of 1. Each convolutional block is followed by:

```text
Batch Normalization
ReLU
Max Pooling
Dropout
```

Global Average Pooling is subsequently used to obtain a 256-dimensional representation, followed by an additional 1-by-1 Conv1D projection to obtain the final \(M\)-dimensional latent representation.

In the experiments reported in the paper, \(M=10\) is used based on the parameter sensitivity analysis.

## Running the Code

The main implementation is provided in:

```text
GP-CNN.py
```

The evaluation procedure is provided in:

```text
factor_eval.py
```

Please ensure that the required Python packages and the corresponding CSI500/CSI1000 data files are available before running the experiments.

Because file paths and computational environments may differ across systems, the relevant data paths should be adjusted according to the local environment before execution.

## Reproducibility

The repository provides the materials required to reproduce and examine the main experiments reported in the paper, including:

- CNN-GP implementation;
- CSI500 experimental data;
- CSI1000 experimental data;
- alpha factor evaluation code;
- supplementary experimental results;
- statistical significance analysis;
- runtime analysis;
- CNN architecture details;
- parameter sensitivity analysis.

For complete experimental settings and additional results, please refer to:

```text
#430-CNN-GP Supplementary Material.pdf
```

## Data Availability

The source code, datasets, and supplementary materials are publicly available in this repository:

https://github.com/202412491495/CNN-GP

## Notes

This repository is provided for academic research and reproducibility purposes.

For detailed descriptions of the CNN-GP framework, experimental settings, evaluation protocol, and additional analyses, please refer to the paper and the supplementary material.
