# TiO₂ Nanoparticle Toxicity — Gene Expression ML Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A machine learning pipeline for analyzing gene expression data to predict and understand TiO₂ nanoparticle toxicity effects. This software was developed as part of PhD research at Mendel University in Brno, Czech Republic (2019–2023), supporting investigations into the antimicrobial and cytotoxic mechanisms of titanium dioxide nanomaterials.

## Features

- **Data Preprocessing**: Handles missing values, scaling, and normalization of gene expression matrices
- **Feature Selection**: Mutual Information scoring and Recursive Feature Elimination (RFE)
- **Model Training**: Support Vector Regression (SVR) with hyperparameter tuning
- **Evaluation**: MSE/RMSE metrics with cross-validation support
- **Visualization**: Publication-ready plots for feature importance and predictions

## Data Source

Gene expression data from NCBI GEO:
- **Accession**: [GSE156564](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156564)
- **Study**: TiO₂ nanoparticle toxicity gene expression profiling

## Installation

```bash
git clone https://github.com/mojo8787/TiO2-Toxicity-ML.git
cd TiO2-Toxicity-ML
pip install -r requirements.txt
```

## Usage

Run the complete analysis pipeline:

```bash
python main.py
```

This executes:
1. Preprocessing of raw gene expression data
2. Feature selection using multiple methods
3. SVR model training with hyperparameter optimization
4. Result visualization and export

## Project Structure

```
├── main.py                    # Main orchestration script
├── preprocessing.py           # Data preprocessing functions
├── feature_selection.py       # Feature selection methods
├── model_training.py          # Model training and evaluation
├── visualization.py           # Result visualization
├── GSE156564_series_matrix.txt # Raw GEO data
└── requirements.txt           # Python dependencies
```

## Results

| Metric | Value |
|--------|-------|
| Mean Squared Error (MSE) | 0.00205 |
| Root Mean Squared Error (RMSE) | 0.0453 |

## Requirements

- Python ≥ 3.8
- scikit-learn
- pandas
- NumPy
- matplotlib
- seaborn

## Citation

If you use this software, please cite:

```bibtex
@software{younis_tio2_toxicity_ml_2023,
  author       = {Younis, Almotasem Bellah},
  title        = {{TiO₂ Nanoparticle Toxicity Gene Expression ML Pipeline}},
  month        = aug,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

## Related Publications

- Younis AB et al. "Synthesis and characterisation of TiO₂–geraniol nanoparticles with synergistic antibacterial activity." *BMC Microbiology*, 2023. DOI: [10.1186/s12866-023-02955-7](https://doi.org/10.1186/s12866-023-02955-7)
- Younis AB et al. "Titanium dioxide nanoparticles: recent progress in antimicrobial applications." *WIREs Nanomedicine & Nanobiotechnology*, 2023. DOI: [10.1002/wnan.1860](https://doi.org/10.1002/wnan.1860)

## Author

**Almotasem Bellah Younis, PhD**  
- ORCID: [0000-0003-2070-2811](https://orcid.org/0000-0003-2070-2811)
- Email: motasem.youniss@gmail.com
- Website: [motasemyounis.com](https://motasemyounis.com)

*Developed during PhD research at Mendel University in Brno, Czech Republic (2019–2023)*

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

This work was supported by the Internal Grant Agency (IGA) of Mendel University (AF-IGA2020-IP068).
