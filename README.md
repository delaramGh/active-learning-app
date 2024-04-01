# Active Image Labeling App (AILA)

This repository contains the code for the Active Image Labeling App (AILA), a tool designed for a specialized labeling task. This task requires the annotator to compare an original image with its altered version.  The "Active" term reflects the active-learning backend that the application has. It reduces the human effort in labeling by using an ML-model to automatically label some of the data based on what the ML-model is learning from the human annotator.


<img src="figs/app.jpg" width="55%">     <img src="figs/app2.jpg" width="40%">


## Getting Started

To use the application, follow these steps:

1. Clone the repository.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Fill the AL-Config.txt file as explained in the next section.
4. Run the script using `streamlit run app.py`

*note that the first run will take some time because the app is creating a .CSV file of the dataset, and doing preprocessing on image pairs.

      
<img src="figs/req.jpg" width="90%">

## Scripts

- `rq1.py`: Script to reproduce the results for RQ1.
- `rq2_dataset.py`: Script to generate the dataset for RQ2.
- `rq2_models.py`: Script to generate the models and evaluate them for RQ2.
- `rq2.py`: Script to reproduce the results for RQ2.

  

## Results

The results are stored in the `output` directory. These results are also presented in the paper. The scripts in this repository can be used to reproduce them.


<img src="figs/out.jpg" width="80%">




