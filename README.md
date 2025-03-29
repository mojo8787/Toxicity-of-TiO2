
# Bioinformatics Machine Learning Project

## Overview

This repository contains machine learning scripts for a bioinformatics project. The project aims to predict gene expression based on various features. The dataset used originates from a study on TiO2 Nanoparticle toxicity.

## Prerequisites

- Python 3.6 or higher
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Data

The dataset used in this project is based on a study of gene expression levels. You'll need to place your dataset in the same directory as the scripts and name it according to the filenames specified in the scripts. The dataset should contain columns for gene expression levels, and various other biological parameters.

## Scripts

1. **preprocessing steps.py**: This script contains code for data preprocessing, including handling missing values, scaling, and normalization.
  
2. **summary statistics after imputation.py**: This script provides summary statistics for the dataset after imputation.
  
3. **Preprocessed_training_data.py**: This is the initial training data preprocessing script.
  
4. **Preprocessed_training_data_2.py**: This is the updated version of the training data preprocessing script.
  
5. **Predections_1.py**: This script is used for making predictions based on the machine learning model.
  
6. **Hyperparameter Tuning.py**: This script is used for hyperparameter tuning of the machine learning model.

## How to Run

1. Make sure you have all the prerequisites installed.
2. Clone the repository to your local machine.
3. Place your dataset in the same directory as the scripts.
4. Run each script in the order specified above.

## Results Interpretation

The machine learning model will output predictions, and performance metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) will be displayed. These metrics can be used to evaluate the performance of the model.

## Citations
(https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156564)
## Contributing

If you'd like to contribute to this project, please fork the repository and make changes as you'd like. Pull requests are warmly welcome.

## Author
Almotasem bellah younis 
- PhD Student in Microbiology at Mendel University in Brno, Czech Republic
# Gene Expression Data Analysis Project

## Overview

This project analyzes gene expression data to identify features that are strong predictors for a specific target variable. The analysis pipeline includes data preprocessing, feature selection, model training, and evaluation.

## Project Structure

```
├── data/                      # Data directory
│   ├── preprocessed_data.xlsx # Preprocessed data (generated)
│   └── GSE156564_series_matrix.txt # Raw data
├── figures/                   # Generated visualizations
├── results/                   # Analysis results
├── main.py                    # Main script orchestrating the analysis
├── preprocessing.py           # Data preprocessing functions
├── feature_selection.py       # Feature selection functions
├── model_training.py          # Model training and evaluation
├── visualization.py           # Result visualization functions
└── requirements.txt           # Python dependencies
```

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script to execute the entire analysis pipeline:

```
python main.py
```

This will:
1. Preprocess the raw gene expression data
2. Select important features using multiple methods
3. Train and evaluate an SVR model with hyperparameter tuning
4. Generate visualizations of the results
5. Save all results to the `results/` directory

## Features

- **Data Preprocessing**: Handles missing values, scales, and normalizes data
- **Feature Selection**: Uses mutual information and Recursive Feature Elimination (RFE)
- **Model Training**: Implements Support Vector Regression with hyperparameter tuning
- **Visualization**: Creates informative plots for analysis and results
- **Results Storage**: Saves all intermediate and final results

## Data Source

The dataset used in this project originates from a study on TiO2 Nanoparticle toxicity:
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156564

## Author

Almotasem bellah younis
- PhD Student in Microbiology at Mendel University in Brno, Czech Republic
