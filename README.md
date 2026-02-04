# AlertDrive: Driver Drowsiness Detection

A comparative analysis project for detecting driver drowsiness from video using multiple machine learning models.

## Project Overview

This project implements and compares multiple approaches for driver drowsiness detection from video data:
- **Baseline Models**: Existing models from research papers and open-source repositories
- **Custom Model**: A newly implemented model for comparison
- **Comparative Analysis**: Performance evaluation across different datasets and metrics

## Project Structure

```
AlertDrive/
├── data/                   # Dataset storage
│   ├── raw/               # Original video datasets
│   ├── processed/         # Preprocessed data
│   └── interim/           # Intermediate processing
├── src/                   # Source code
│   ├── models/           
│   │   ├── baseline/     # Existing model implementations
│   │   └── custom/       # Custom model implementation
│   ├── preprocessing/    # Data preprocessing
│   ├── evaluation/       # Evaluation metrics
│   ├── visualization/    # Plotting utilities
│   └── utils/            # Helper functions
├── notebooks/            # Jupyter notebooks
│   ├── exploratory/     # Data exploration
│   └── experiments/     # Model experiments
├── trained_models/      # Saved model weights
├── results/             # Experiment results
│   ├── metrics/        # Performance metrics
│   ├── figures/        # Visualizations
│   ├── reports/        # Analysis reports
│   └── comparisons/    # Comparative analysis
├── configs/            # Configuration files
├── tests/              # Unit tests
├── docs/               # Documentation
└── logs/               # Training logs

```

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Key Features

- **Reproducibility**: Properly implement and evaluate existing models
- **Dataset Diversity**: Test on multiple datasets with varying characteristics
- **Comparative Analysis**: Consistent evaluation metrics (accuracy, F1, precision, recall)
- **Comprehensive Results**: Detailed performance comparison and insights

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC
- Processing Time

## Usage

(To be filled in during implementation)

## Contributors

(Add your team members)

## License

(Add license information)
