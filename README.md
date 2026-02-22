 # 📈 LSTM Stock Price Prediction

A comprehensive deep learning framework for stock price prediction using LSTM neural networks with advanced feature engineering, automated hyperparameter optimization, and interactive trading simulation.

## ✨ Features

- **LSTM-Based Time Series Forecasting**: LSTM specifically designed for sequential financial data
- **Automated Hyperparameter Optimization**: Grid search across multiple hyperparameters with early stopping
- **Automated Data Retrieval Pipeline**: yfinance and fredapi integration to retrieve up-to-date market and economic data
- **Feature Engineering Pipeline**: Technical indicators (MACD, RSI, ATR) and macroeconomic data integration
- **Interactive Dashboard**: Real-time experiment configuration with Feature Selection and Hyperparameter Search Configuration in a visual Dashboard UI
- **Trading Strategy Simulation**: Backtest several different trading strategies with detailed performance metrics
- **Comprehensive Visualization**: Training progress, predictions, error analysis, and equity curves
- **Flexible Model Persistence**: Save and load trained models with configurations

## 🏗️ Architecture

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)
- PyTorch with CUDA support

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LSTMPrediction
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Setup

Place your financial data in the `data/` directory. Expected format:
- CSV files with date index
- Columns: `Close_{TICKER}`, `Volume_{TICKER}`, technical indicators, macroeconomic data

OR retrieve data using the provided pipeline:

```
data_acquision.ipynb
```

## 📊 Usage

### 1. Quick Start with Jupyter Notebook

Open `StockPricePrediction.ipynb` and run through the cells:

### 2. Interactive Dashboard

Launch the interactive dashboard for experiment configuration:

Configure:
- **Training Features**: Select from technical indicators and macro features
- **Hyperparameter Grid**: Window size, batch size, learning rate, dropout, etc.
- **Model Architecture**: Hidden layers, LSTM layers, optimizer selection

### 3. Feature Selection

The framework supports incremental feature selection:

This tests:
1. Target feature only (baseline)
2. Target + each additional feature individually
3. All features combined

### 4. Model Evaluation

Generates:
- Training history plots
- Predictions vs actuals on test set
- Residual analysis
- Performance metrics (MSE, MAE, R²)

### 5. Trading Strategy Backtesting


## 🔧 Configuration

## 📈 Supported Features

### Technical Indicators
- Price data: `Close`, `Volume`
- Momentum: `MACD`, `RSI`
- Volatility: `ATR`, `VIX`

### Macroeconomic Data
- `MACRO_cpi`: Consumer Price Index
- `MACRO_rate`: Interest rates
- `MACRO_ppi`: Producer Price Index
- `MACRO_gdp`: GDP
- `MACRO_unrate`: Unemployment rate
- `MACRO_CSI`: Consumer Sentiment Index
- `MACRO_HS`: Housing Starts

## 🎯 Model Training Pipeline

1. **Data Preparation**: Load, clean, and scale data
2. **Feature Engineering**: Calculate technical indicators
3. **Sequence Creation**: Generate sliding windows for LSTM input
4. **Train/Val/Test Split**: 80%/10%/10% split
5. **Grid Search**: Automated hyperparameter optimization
6. **Early Stopping**: Prevent overfitting with validation monitoring
7. **Model Selection**: Choose best model based on validation loss
8. **Evaluation**: Test set performance and visualization
9. **Trading Simulation**: Backtest with realistic trading scenarios

## 📁 Project Structure

```
LSTMPrediction/
├── classes/              # Core framework classes
├── data/                 # Financial datasets
├── run/                  # Experiment outputs and checkpoints
├── StockPricePrediction.ipynb  # Main training notebook
├── data_acquisition.ipynb      # Data collection pipeline
├── simulation.ipynb            # Trading simulations
└── requirements.txt            # Python dependencies
```

## 🔬 Experiment Tracking

All experiments are automatically saved to `run/` with:
- Model checkpoints (`checkpoint.pt`)
- Configuration files (`config.json`)
- Training plots
- Prediction visualizations
- Performance metrics CSV

## 🎨 Visualization

The framework provides comprehensive visualizations:

- **Training Progress**: Real-time loss curves with early stopping indicators
- **Predictions**: Overlaid predicted vs actual values on test set
- **Residual Analysis**: Error distribution and patterns
- **Equity Curves**: Trading simulation performance over time
- **Feature Analysis**: SHAP values for feature importance