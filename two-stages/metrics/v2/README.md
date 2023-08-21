## Models

---- version.1

### Linear Stack - Base Models

- Linear Regression
- VAR
- ElasticNet

### Tree Stack - Base Models

- LightGBM
- XGBoost
- Gradient Boosting Machine

### Neural Network Stack - Base Models

- LSTM
- CNNs
- RNNs

### Meta Models

- Linear Stack: Ridge Regression
- Tree Stack: Random Forest
- Neural Network Stack: Linear Layer Neural Network

--- version.2: Share the same base models

### Base Models

Traditional Statistical Models: These models are built on classic time series methodologies.

ARIMA: As mentioned, it stands for AutoRegressive Integrated Moving Average.
SARIMA: Seasonal ARIMA.
Exponential Smoothing (ETS): There are several types including Simple Exponential Smoothing, Double Exponential Smoothing (Holt's linear method), and Triple Exponential Smoothing (Holt-Winters method).
GARCH (Generalized Autoregressive Conditional Heteroskedasticity): Often used for financial time series to model volatility.
UMA (Unobserved Components Models): This model decomposes time series into components like trend and seasonality.
Prophet:

Essentially, there's just the Prophet model developed by Facebook. However, you can modify and extend it in various ways by adding holidays, incorporating external regressors, adjusting seasonality, etc.
Neural Networks:

Simple RNN: The basic recurrent neural network structure.
LSTM: Long Short-Term Memory network.
GRU: Gated Recurrent Unit.
Attention Mechanisms: Used in combination with RNNs or LSTMs to give weighted importance to different time steps.
Transformer Architectures: Including models like BERT, GPT, T5, and Vaswani Transformer. While they originated in NLP, they can be adapted for time series.
1D Convolutional Neural Networks (1D CNNs): They can be used for time series prediction by viewing the series as a one-dimensional "image."
TCN (Temporal Convolutional Network): Uses causal convolutions for sequence modeling.
Bayesian Time Series Models:

Bayesian Structural Time Series (BSTS): Allows for model specification using components like local linear trends, seasonality, and holidays.
Gaussian Processes (GPs): A non-parametric Bayesian approach to time series forecasting.
Kalman Filters: Recursive models that estimate the state of a linear dynamic system from a series of noisy measurements.
Others:

Dynamic Time Warping (DTW): As previously mentioned, primarily for time series comparison.
VAR (Vector Autoregression): And its extensions like VARMA (Vector Autoregressive Moving-Average) and VARMAX (with exogenous variables).
cointegration models: Such as VECM (Vector Error Correction Model), often used in econometrics to find long-term relationships between time series.
State Space Models: A broad category including various models that represent time series as a series of state transitions.
Hidden Markov Models (HMMs): Probabilistic models where the system being modeled is assumed to be a Markov process with unobserved states.

### Linear Stack - Meta Model

- Linear Regression

### Classification Stack - Meta Model

- Logistic Regression

### Deep Learning Stack - Meta Model

- Linear Layer Neural Network
