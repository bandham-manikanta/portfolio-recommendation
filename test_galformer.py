import pandas as pd
import yfinance as yf
import pandas_ta as ta
from fredapi import Fred
from datetime import datetime, timedelta
import holidays
import numpy as np
import tensorflow as tf

# Initialize Fred with your API key
fred = Fred(api_key='db61e0d65c4d2a1053221aec21822d4e')  # Replace with your actual API key securely

# Define economic indicators and their series IDs
indicators = {
    'Effective Federal Funds Rate': 'FEDFUNDS',
    '10-Year Treasury Rate': 'DGS10',
    'Consumer Price Index': 'CPIAUCSL',
    'Producer Price Index': 'PPIACO',
    'Unemployment Rate': 'UNRATE',
    'Nonfarm Payroll Employment': 'PAYEMS',
    'Real GDP': 'GDPC1',
    'Housing Starts': 'HOUST',
    'Industrial Production Index': 'INDPRO',
    'M2 Money Stock': 'M2SL',
    'Crude Oil Prices': 'DCOILWTICO',
    'Retail Sales': 'RSXFS',
    'Total Business Inventories': 'BUSINV'
}

def fetch_economic_data(indicators, end_date):
    # Fetch data up to today and use a wide enough range to cover historical needs
    start_date = (end_date - timedelta(days=365)).strftime('%Y-%m-%d')  # One year of history
    economic_data = pd.DataFrame()
    for name, series_id in indicators.items():
        try:
            data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date.strftime('%Y-%m-%d'))
            if data is not None and not data.empty:
                economic_data[name] = data
                print(f"Successfully fetched data for {name}")
        except Exception as e:
            print(f"Error fetching {name}: {e}")
    economic_data.index = pd.to_datetime(economic_data.index)
    economic_data.sort_index(inplace=True)
    return economic_data

def fetch_stock_data(ticker, end_date):
    # Fetch stock data for the past year up to the current date
    start_date = end_date - timedelta(days=365)
    stock_data = yf.download(
        ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
        progress=False,
        group_by='ticker'
    )
    if stock_data.empty:
        print(f"No stock data fetched for {ticker} between {start_date.date()} and {end_date.date()}")
        return None
    stock_data.index = pd.to_datetime(stock_data.index)
    # Flatten columns if MultiIndex
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(1)
    return stock_data

def calculate_technical_indicators(data, ticker):
    close_col = 'Close'
    if close_col in data.columns:
        data[f'{ticker}_SMA_20'] = ta.sma(data[close_col], length=20)
        data[f'{ticker}_RSI_14'] = ta.rsi(data[close_col], length=14)
        macd = ta.macd(data[close_col], fast=12, slow=26)
        if macd is not None and not macd.empty:
            macd_columns = [f'{ticker}_MACD', f'{ticker}_MACD_Hist', f'{ticker}_MACD_Signal']
            macd.columns = macd_columns
            data = pd.concat([data, macd], axis=1)
        else:
            print(f"Failed to calculate MACD for {ticker}")
        bbands = ta.bbands(data[close_col], length=20)
        if bbands is not None and not bbands.empty:
            bbands_columns = [f'{ticker}_BB_Lower', f'{ticker}_BB_Middle', f'{ticker}_BB_Upper', f'{ticker}_BB_Bandwidth', f'{ticker}_BB_Percentage']
            bbands.columns = bbands_columns
            data = pd.concat([data, bbands], axis=1)
        else:
            print(f"Failed to calculate Bollinger Bands for {ticker}")
        data[f'{ticker}_MOM_10'] = ta.mom(data[close_col], length=10)
    else:
        print(f"Missing 'Close' column in stock data for {ticker}")
    return data

def create_lag_features(data, indicators, ticker, n_lags=5):
    lagged_features = {}
    # Create lag features for stock prices
    close_col = 'Close'
    if close_col in data.columns:
        for lag in range(1, n_lags + 1):
            lag_col_name = f'{ticker}_Close_Lag_{lag}'
            lagged_features[lag_col_name] = data[close_col].shift(lag)
    else:
        print(f"Column {close_col} not found in stock data.")
    # Create lag features for economic indicators
    for indicator in indicators:
        if indicator in data.columns:
            for lag in range(1, n_lags + 1):
                lag_col_name = f'{indicator}_Lag_{lag}'
                lagged_features[lag_col_name] = data[indicator].shift(lag)
        else:
            print(f"Indicator {indicator} not found in combined_data.")
    lagged_features_df = pd.DataFrame(lagged_features, index=data.index)
    data = pd.concat([data, lagged_features_df], axis=1)
    return data

def add_date_features(data):
    data['Day_of_Week'] = data.index.dayofweek  # Monday=0, Sunday=6
    data['Month'] = data.index.month
    data['Quarter'] = data.index.quarter
    us_holidays = holidays.US()
    data['Is_Holiday'] = data.index.isin(us_holidays).astype(int)
    data['Is_Month_Start'] = data.index.is_month_start.astype(int)
    data['Is_Month_End'] = data.index.is_month_end.astype(int)
    return data

def prepare_input_features(stock_data, economic_data, ticker):
    # Ensure indices are single-level and have the same name
    if stock_data.index.nlevels > 1:
        stock_data.reset_index(inplace=True)
        stock_data.set_index('Date', inplace=True)
    if economic_data.index.nlevels > 1:
        economic_data.reset_index(inplace=True)
        economic_data.set_index('Date', inplace=True)
    stock_data.index.name = 'Date'
    economic_data.index.name = 'Date'
    # Merge stock data and economic data
    combined_data = pd.merge(stock_data, economic_data, left_index=True, right_index=True, how='left')
    # Fill missing values
    combined_data.fillna(method='ffill', inplace=True)
    # Calculate technical indicators
    combined_data = calculate_technical_indicators(combined_data, ticker)
    # Create lag features
    combined_data = create_lag_features(combined_data, indicators.keys(), ticker)
    # Add date-based features
    combined_data = add_date_features(combined_data)
    return combined_data

def positional_encoding(sequence_length, d_model):
    position = np.arange(sequence_length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000.0, (2 * (i//2)) / np.float32(d_model))
    angle_rads = position * angle_rates
    sin_terms = np.sin(angle_rads[:, 0::2])
    cos_terms = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.zeros((sequence_length, d_model))
    pos_encoding[:, 0::2] = sin_terms
    pos_encoding[:, 1::2] = cos_terms
    return pos_encoding

def add_positional_encoding(inputs):
    pe = positional_encoding(inputs.shape[1], inputs.shape[2])
    pe = np.expand_dims(pe, axis=0)  # Shape: (1, sequence_length, num_features)
    inputs_with_pe = inputs + pe
    return inputs_with_pe

def main():
    ticker = 'AAPL'
    today = pd.Timestamp.today().normalize()
    
    # Fetch economic data
    economic_data = fetch_economic_data(indicators, today)
    if economic_data.empty:
        print("Economic data is empty.")
        return

    # Fetch stock data
    stock_data = fetch_stock_data(ticker, today)
    if stock_data is None:
        return

    # Prepare input features
    combined_data = prepare_input_features(stock_data, economic_data, ticker)

    # Now, prepare the input sequence for the model

    # Ensure the DataFrame is sorted by date
    combined_data.sort_index(inplace=True)

    # Define required columns
    required_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        f'{ticker}_SMA_20', f'{ticker}_RSI_14', f'{ticker}_MACD', 
        f'{ticker}_MACD_Hist', f'{ticker}_MACD_Signal',
        f'{ticker}_BB_Lower', f'{ticker}_BB_Middle', f'{ticker}_BB_Upper',
        f'{ticker}_BB_Bandwidth', f'{ticker}_BB_Percentage',
        f'{ticker}_MOM_10'
    ] + [f'{ticker}_Close_Lag_{i}' for i in range(1, 6)] + \
        [f'{ind}_Lag_{i}' for ind in indicators.keys() for i in range(1, 6)] + \
        ['Day_of_Week', 'Month', 'Quarter', 'Is_Holiday', 'Is_Month_Start', 'Is_Month_End']

    # Check for missing columns
    missing_cols = [col for col in required_columns if col not in combined_data.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return

    # Prepare input sequence
    sequence_length = 60  # Should match the sequence length used during training
    if len(combined_data) >= sequence_length:
        input_sequence = combined_data[required_columns].tail(sequence_length).values  # Shape (sequence_length, num_features)
        # Check for NaNs
        if np.isnan(input_sequence).any():
            print("Input sequence contains NaN values after feature calculation.")
            return
        # Normalize features (per sample as during training)
        feature_mean = np.mean(input_sequence, axis=0, keepdims=True)
        feature_std = np.std(input_sequence, axis=0, keepdims=True) + 1e-6
        normalized_input_sequence = (input_sequence - feature_mean) / feature_std
        # Reshape to (1, sequence_length, num_features)
        normalized_input_sequence = normalized_input_sequence.reshape(1, sequence_length, -1).astype(np.float32)
    else:
        print(f"Not enough data to form input sequence of length {sequence_length}")
        return

    # Add positional encoding
    inputs_with_pe = add_positional_encoding(normalized_input_sequence)

    # Load the trained model
    model = tf.keras.models.load_model('enhanced_stock_galformer_model.keras')
    print("Model loaded successfully.")

    # Make prediction
    predictions = model.predict(inputs_with_pe)
    print(f"Predictions (standardized) shape: {predictions.shape}")

    # Since labels were normalized during training per sample, and during inference we don't have label mean and std,
    # these predictions are in standardized form and cannot be directly denormalized.
    # If you have global label_mean and label_std from training data, use them here to denormalize.
    # For now, we will print the standardized predictions.

    print("Predictions (standardized):")
    print(predictions.flatten())

if __name__ == "__main__":
    main()
