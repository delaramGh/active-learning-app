# Active Image Labeling App (AILA)

This repository contains the code for the Active Image Labeling App (AILA), a tool designed for a specialized labeling task. This task requires the annotator to compare an original image with its altered version.  The "Active" term reflects the active-learning backend that the application has. It reduces the human effort in labeling by using an ML-model to automatically label some of the data based on what the ML-model is learning from the human annotator.

<img src="figs/fig1.jpg" width="900">

## Getting Started

To reproduce the results in the paper, follow these steps:

1. Clone the repository.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the scripts in the following order: (Don't forget to put the dataset csv files beside the scripts)
   1. `rq1.py`
   2. `rq2_dataset.py`
   3. `rq2_models.py`
   4. `rq2.py`

## Scripts

- `rq1.py`: Script to reproduce the results for RQ1.
- `rq2_dataset.py`: Script to generate the dataset for RQ2.
- `rq2_models.py`: Script to generate the models and evaluate them for RQ2.
- `rq2.py`: Script to reproduce the results for RQ2.

## Results

The results are stored in the `output` directory. These results are also presented in the paper. The scripts in this repository can be used to reproduce them.




