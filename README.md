# ML-OOPS Healthcare

An integrated healthcare system for pneumonia detection that combines deep learning-based chest X-ray analysis with clinical data to provide comprehensive medical assessments.

## Overview

This system uses state-of-the-art machine learning techniques to analyze chest X-ray images for pneumonia detection while incorporating patient clinical data for more accurate diagnoses. It features both production and uncertainty models to provide confidence levels in predictions.

## Features

- Deep learning-based chest X-ray analysis using DenseNet121
- Uncertainty quantification for predictions
- Integration of clinical data with image analysis
- Risk level assessment and clinical recommendations
- Interactive web interface using Chainlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/realjules/ml_oops_healthcare.git
cd ml_oops_healthcare
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The system can be used through the web interface powered by Chainlit. To start the server:

```bash
chainlit run app.py
```

## Project Structure

- `app.py` - Web interface implementation using Chainlit
- `model.py` - Core ML model implementation and integrated analysis system
- `cmu_dash.py` - Dashboard implementation
- `demo_dataset/` - Sample dataset for testing
- `test/` - Test files and utilities

## Technical Details

The system uses a DenseNet121 architecture modified for grayscale image input, with two main components:
1. A production model for primary predictions
2. An uncertainty model using Monte Carlo Dropout for Bayesian uncertainty estimation:
   - Performs 30 forward passes with dropout enabled during inference
   - Calculates prediction variance and confidence intervals
   - Uses 30% dropout rate for robust uncertainty sampling

The system combines these predictions with patient clinical data to generate:
- Pneumonia probability scores
- Uncertainty measurements (standard deviation and confidence intervals)
- Risk level assessments
- Clinical recommendations

The Monte Carlo Dropout approach allows the model to estimate its own uncertainty, which is crucial for medical applications where knowing the confidence level of predictions can impact clinical decision-making.