# Active Image Labeling App (AILA)

This repository contains the code for the Active Image Labeling App (AILA), a tool designed for a specialized labeling task. This task requires the annotator to compare an original image with its altered version.  The "Active" term reflects the active-learning backend that the application has. It reduces the human effort in labeling by using an ML-model to automatically label some of the data based on what the ML-model is learning from the human annotator.

## Abstract

This paper presents the first comprehensive study on test flakiness for simulation-based testing of autonomous driving systems (ADS). Simulators are widely used for testing and verification of ADS.
However, simulators might be flaky leading to inconsistent test results. We aim to answer two research questions in this paper: (1)~How do flaky ADS simulations impact automated testing that relies on randomized algorithms? and (2)~Can machine learning (ML) accurately identify flaky ADS tests in a cost-effective manner, while requiring few or no re-executions? Our empirical results obtained based on two widely-used open-source ADS simulators and five different ADS test setups demonstrate that the occurrence of test flakiness for ADS is at least as prevalent as test flakiness observed in code-bases. Further, flaky tests significantly impact randomized and stochastic testing with respect to two widely-used metrics: the number of distinct failures detected and the severity of the failures detected. Finally, ML classifiers effectively identify flaky ADS tests by requiring only a minimal cost of a single execution for each test. They achieve a recall of at least $76$\% and a precision of at least $89$\%, significantly outperforming a non-ML baseline.

## Folder Structure

```bash
.
├── data
├── output
│   ├── rq1-1
│   ├── rq1-2
│   ├── rq1-3
│   ├── rq2-1
│   └── rq2-2
├── scripts
├── setups
│   ├── beamng
│   ├── comp
│   └── pid
└── supplementary_materials
```

- `data`: Contains the datasets used in the paper.
- `output`: Contains outputs of the scripts organized by research questions (RQs) and their sub-questions.
- `scripts`: Contains the scripts used to generate the results in the paper.
- `setups`: Contains the setups used in the paper.

## Datasets

- `ds_pid.csv`: Dataset for the PID ADS.
- `ds_pylot.csv`: Dataset for the Pylot ADS.
- `ds_transfuser.csv`: Dataset for the Transfuser ADS.
- `ds_beamng.csv`: Dataset for the BeamNG ADS.
- `ds_beamng_competition.csv`: Dataset for the BeamNG Competition ADS.

## Scripts

- `rq1.py`: Script to reproduce the results for RQ1.
- `rq2_dataset.py`: Script to generate the dataset for RQ2.
- `rq2_models.py`: Script to generate the models and evaluate them for RQ2.
- `rq2.py`: Script to reproduce the results for RQ2.

## Results

The results are stored in the `output` directory. These results are also presented in the paper. The scripts in this repository can be used to reproduce them.

## Supplementary Materials

The supplementary materials for the paper containing additional results (plots, tables, etc.) for different thresholds are stored in the `supplementary_materials` directory.


## Getting Started

To reproduce the results in the paper, follow these steps:

1. Clone the repository.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the scripts in the following order: (Don't forget to put the dataset csv files beside the scripts)
   1. `rq1.py`
   2. `rq2_dataset.py`
   3. `rq2_models.py`
   4. `rq2.py`


