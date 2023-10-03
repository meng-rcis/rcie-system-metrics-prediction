# rcie-system-metrics-prediction

## Command to run jupyter notebook

### Windows

```bash
cd D:\nuttchai\dev\Projects\rcie\repo\rcie-system-metrics-prediction; jupyter notebook
```

## Note - v1 (Archive)

1. Single input, single output - SISO (Univariate time series forecasting): Here, you're using past values of a single time series to predict future values of the same time series.

2. Multiple inputs, single output - MISO (Multivariate time series forecasting): In this case, you're using past values from multiple time series to predict future values of a single target time series.

3. Multiple inputs, multiple outputs - MIMO (Multi-step and multivariate time series forecasting): You're using past values from multiple time series to predict multiple future time steps of all those time series. Note that this setup is quite complex and might be challenging to manage and interpret.

There are several models that can handle multivariate input for single output prediction, especially in the context of time series forecasting. Here are a few examples:

Long Short-Term Memory (LSTM) Models: As we've already discussed, LSTM models are a type of recurrent neural network (RNN) that can handle sequences of data. They're ideal for time series forecasting because they have a sense of "memory" and can use this to make sense of the current input in the context of what's been seen before.

Convolutional Neural Networks (CNN): CNNs are generally used in image processing but they can be very useful for time series prediction too. A 1D CNN is capable of capturing patterns within a window of consecutive time steps and can therefore be trained to find patterns over time. This makes them useful for multivariate time series forecasting.

Vector Auto Regression (VAR): VAR is a type of model that can predict multiple output variables using linear combinations of past values of all variables in the system. However, it doesn't capture complex and non-linear relationships between variables as well as LSTM and CNN models do.

Prophet: Prophet, developed by Facebook, is a model that can handle multivariate time series data. It's less flexible than LSTM and CNN models because it makes certain assumptions about your data, but it can be a good choice if your time series have strong seasonal effects and you want a model that's easy to understand and interpret.

AutoRegressive Integrated Moving Average (ARIMA): ARIMA models are popular for time series forecasting. The ARIMA model makes predictions based on the idea that future values are a function of past values and errors. However, traditional ARIMA models are univariate, and can't handle multiple input variables. The multivariate version of ARIMA, known as Vector Autoregressive Moving Average (VARMA) or Multivariate ARIMA (MARIMA), can be used but they are more complex and computationally expensive.

The usage of n_past and n_future is not specific to LSTM, it's more about a way of framing your time series forecasting problem. It's often called a sliding window approach where you use a window of past observations (n_past) to predict a number of future steps (n_future).

So yes, you can apply a similar approach for creating your input/output pairs in other models such as CNNs, ARIMA, VAR, or Prophet. The main difference will be in how you structure your data and how you use these models. For example:

For CNNs, your data structure will be similar to LSTM. The difference is in the model architecture, where you use convolutional layers instead of LSTM layers.

For ARIMA, the data structure is usually univariate and the model predicts one step at a time. For multivariate version like VARMA, you would still structure your data in a similar way with a window of past observations.

For Prophet, it is typically used for univariate time series forecasting, but additional regressors (similar to additional features in a multivariate forecast) can be added. The main input to Prophet is a DataFrame with a date-time column and one or more numeric columns. Prophet uses this data to fit its model and make predictions.

VAR is specifically designed for multivariate time series. It uses a set number of previous time steps from all variables in the system to predict the next time step.

## Note - v2 (current)

### Project Structure

- To-do: Add later on

### Project Directory

Main file located at `models/predictions/features/main.py`

- Next Action

Setup Manager 
- we create training data to train meta model 
- if prediction_steps = 3 ( training data: predict_l1n1, predict_l1n2, predict_l1n3, predict_l2n1, predict_l2n2, predict_l2n3, predict_l3n1, ... )

Main Manager 
- if prediction_steps = 3, current loop 3 ( training data: predict_l1n1, predict_l2n1, predict_l3n1, predict_l3n2, predict_l3n3 [latest] )
- if trainning dataset be like that ( training data: predict_l1n1, predict_l2n1, predict_l3n1, predict_l3n2, predict_l3n3 [latest] ), is it possible to have two dataset?
  
( training data - to train meta : predict_l1n1, predict_l1n2, predict_l1n3, predict_l2n1, predict_l2n2, predict_l2n3, predict_l3n1, ... )

( training data - to collect after meta predict - only used to display dashboard : ~~predict_l1n1, predict_l2n1,~~ predict_l3n1, predict_l3n2, predict_l3n3 [latest] )
