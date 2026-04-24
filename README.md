# SmartEx Model

A machine learning pipeline for textile data spectral analysis with multiple model architectures, dimensionality reduction, and MLflow experiment tracking.

## Main Functionalities

- **Autoencoder Models**: Spectral autoencoders with residual blocks and contrastive learning support
- **Classification Chains**: Multiple classifier chain architectures including:
  - Logistic Regression
  - SVM (Support Vector Machine)
  - XGBoost
  - Multi-Layer Perceptron (MLP)
- **Data Processing**: Hyperspectral image loading, preprocessing, and multi-level evaluation (pixel, line, cube)
- **Experiment Tracking**: MLflow integration for logging parameters, metrics, and artifacts
- **Hyperparameter Sweeping**: Automated grid search over model and data configurations

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your dataset paths and experiment settings in `config_ae.yaml`

3. Set up MLflow tracking URI (optional, required for remote experiment tracking)

## Entry Points

- **`sweep_ae.py`**: Launch hyperparameter sweep experiments. Creates multiple training runs based on grid configurations
- **`create_gkf.py`**: Generate training/evaluation splits and create model configurations

## Usage

Run a hyperparameter sweep:
```bash
python sweep_ae.py
```

For detailed experiment tracking and logging, configure your MLflow tracking URI in `config_ae.yaml`.
