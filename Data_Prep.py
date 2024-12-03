# Import necessary libraries
import pandas as pd
from fredapi import Fred
import numpy as np
import holidays
import pandas_ta as ta
import os

# Set display options for pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# Initialize Fred with your API key
fred = Fred(api_key='YOUR_FRED_API_KEY')  # Replace with your actual API key securely

# Define date range
start_date = '2002-01-01'
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')  # Use today's date

# Define the indicators and their series IDs
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

# Fetch the economic data within the specified date range
economic_data = pd.DataFrame()
for name, series_id in indicators.items():
    try:
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        if data is not None and not data.empty:
            economic_data[name] = data
            print(f"Successfully fetched data for {name}")
    except Exception as e:
        print(f"Error fetching {name}: {e}")

# Convert index to datetime and sort
economic_data.index = pd.to_datetime(economic_data.index)
economic_data.sort_index(inplace=True)

# Function to fill missing values in economic data
def fill_missing_values(df):
    df_filled = df.copy()

    # Ensure index is DatetimeIndex
    if not isinstance(df_filled.index, pd.DatetimeIndex):
        df_filled.index = pd.to_datetime(df_filled.index)
    df_filled.sort_index(inplace=True)

    # Create Month and Year columns
    df_filled['Month'] = df_filled.index.month
    df_filled['Year'] = df_filled.index.year

    # Process each column individually
    for column in df.columns:
        col_data = df_filled[['Year', 'Month', column]].copy()

        # Calculate monthly means
        monthly_means = col_data.groupby(['Year', 'Month'])[column].mean().rename('Monthly_Mean').reset_index()

        # Merge monthly means back into col_data
        col_data = col_data.merge(monthly_means, on=['Year', 'Month'], how='left')

        # Fill missing values with Monthly Mean
        null_mask = col_data[column].isnull()
        col_data.loc[null_mask, column] = col_data.loc[null_mask, 'Monthly_Mean']

        # Calculate yearly means
        yearly_means = col_data.groupby('Year')[column].mean().rename('Yearly_Mean').reset_index()

        # Merge yearly means into col_data
        col_data = col_data.merge(yearly_means, on='Year', how='left')

        # Fill remaining missing values with Yearly Mean
        still_null_mask = col_data[column].isnull()
        col_data.loc[still_null_mask, column] = col_data.loc[still_null_mask, 'Yearly_Mean']

        # Update the filled values back into df_filled
        df_filled.loc[:, column] = col_data[column].values

    # Drop the auxiliary columns
    df_filled.drop(['Month', 'Year'], axis=1, inplace=True)

    return df_filled

# Apply the function to fill missing values
economic_data_filled = fill_missing_values(economic_data)
economic_data_filled.index = pd.to_datetime(economic_data_filled.index)
economic_data_filled.sort_index(inplace=True)

# Read stock data from a Parquet file
stock_data = pd.read_parquet('sp500_50stocks_data.parquet')  # Ensure this file exists in your directory
stock_data.index = pd.to_datetime(stock_data.index)
stock_data.sort_index(inplace=True)

# Flatten MultiIndex columns in stock_data
stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]

# Merge economic data with stock data
economic_data_filled.index.name = 'Date'
stock_data.index.name = 'Date'

# Create a daily date range based on stock data index
daily_date_range = pd.date_range(start=stock_data.index.min(), end=stock_data.index.max(), freq='D')

# Reindex economic data to daily frequency using forward fill
economic_data_daily = economic_data_filled.reindex(daily_date_range)
economic_data_daily.index.name = 'Date'
economic_data_daily.fillna(method='ffill', inplace=True)

# Merge the DataFrames using the date index
combined_data = stock_data.join(economic_data_daily, how='left')
combined_data.fillna(method='ffill', inplace=True)

# List of tickers to process
tickers = ["AAPL", "NVDA", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "AVGO", "LLY", "TSLA",
           "WMT", "JPM", "V", "XOM", "UNH", "ORCL", "MA", "HD", "PG", "COST", "JNJ",
           "NFLX", "ABBV", "BAC", "KO", "CRM", "CVX", "MRK", "TMUS", "AMD", "PEP",
           "ACN", "LIN", "TMO", "MCD", "CSCO", "ADBE", "WFC", "IBM", "GE", "ABT",
           "DHR", "AXP", "MS", "CAT", "NOW", "QCOM", "PM", "ISRG", "VZ"]

print("Initial number of columns:", len(combined_data.columns))

# Calculate technical indicators for each ticker
for ticker in tickers:
    close_col = f'{ticker}_Close'

    if close_col in combined_data.columns:
        # Calculate technical indicators
        combined_data[f'{ticker}_SMA_20'] = ta.sma(combined_data[close_col], length=20)
        combined_data[f'{ticker}_RSI_14'] = ta.rsi(combined_data[close_col], length=14)
        macd = ta.macd(combined_data[close_col], fast=12, slow=26)
        macd_columns = [f'{ticker}_MACD', f'{ticker}_MACD_Hist', f'{ticker}_MACD_Signal']
        if macd is not None and not macd.empty:
            macd.columns = macd_columns
            combined_data = pd.concat([combined_data, macd], axis=1)
        bbands = ta.bbands(combined_data[close_col], length=20)
        bbands_columns = [
            f'{ticker}_BB_Lower', f'{ticker}_BB_Middle', f'{ticker}_BB_Upper',
            f'{ticker}_BB_Bandwidth', f'{ticker}_BB_Percentage'
        ]
        if bbands is not None and not bbands.empty:
            bbands.columns = bbands_columns
            combined_data = pd.concat([combined_data, bbands], axis=1)
        combined_data[f'{ticker}_MOM_10'] = ta.mom(combined_data[close_col], length=10)
    else:
        print(f'Column {close_col} not found in combined_data.')

print("Number of columns after adding technical indicators:", len(combined_data.columns))

# Create lag features
n_lags = 5
lagged_features = {}

# Create lag features for stock prices
for ticker in tickers:
    close_col = f'{ticker}_Close'

    if close_col in combined_data.columns:
        for lag in range(1, n_lags + 1):
            lag_col_name = f'{ticker}_Close_Lag_{lag}'
            lagged_features[lag_col_name] = combined_data[close_col].shift(lag)
    else:
        print(f'Column {close_col} not found in combined_data.')

# Create lag features for economic indicators
economic_indicators = economic_data_filled.columns.tolist()

for indicator in economic_indicators:
    if indicator in combined_data.columns:
        for lag in range(1, n_lags + 1):
            lag_col_name = f'{indicator}_Lag_{lag}'
            lagged_features[lag_col_name] = combined_data[indicator].shift(lag)
    else:
        print(f'Indicator {indicator} not found in combined_data.')

# Convert the lagged features dictionary to a DataFrame
lagged_features_df = pd.DataFrame(lagged_features, index=combined_data.index)

# Concatenate the lagged features DataFrame to the original DataFrame
combined_data = pd.concat([combined_data, lagged_features_df], axis=1)
combined_data = combined_data.copy()

print(f"Total number of columns after adding lag features: {len(combined_data.columns)}")

# Extract Date-Based Features
combined_data['Day_of_Week'] = combined_data.index.dayofweek  # Monday=0, Sunday=6
combined_data['Month'] = combined_data.index.month
combined_data['Quarter'] = combined_data.index.quarter
us_holidays = holidays.US()
combined_data['Is_Holiday'] = combined_data.index.isin(us_holidays).astype(int)
combined_data['Is_Month_Start'] = combined_data.index.is_month_start.astype(int)
combined_data['Is_Month_End'] = combined_data.index.is_month_end.astype(int)

# Split combined data into individual company DataFrames
all_dfs = {}  # Dictionary to store DataFrames

for tick in tickers:
    # Create a DataFrame for each company
    df_name = "df_" + tick

    # Filter the combined_data DataFrame for columns matching the ticker
    company_df = combined_data.filter(like=tick, axis=1)

    # Add date-based features and economic indicators
    company_df = pd.concat([
        company_df,
        combined_data[['Day_of_Week', 'Month', 'Quarter', 'Is_Holiday', 'Is_Month_Start', 'Is_Month_End']],
        combined_data[economic_indicators]
    ], axis=1)

    # Add target variables
    for i in range(1, 6):  # Forwarded columns for 1 to 5 steps
        company_df[f'{tick}_target_{i}'] = company_df[f'{tick}_Close'].shift(-i)

    # Drop missing values
    company_df.dropna(inplace=True)

    # Store the DataFrame in the dictionary
    all_dfs[df_name] = company_df

    # Save company DataFrame to CSV
    csv_filename = f'{df_name}.csv'
    company_df.to_csv(csv_filename)
    print(f"Saved {df_name} to {csv_filename}")

# Adjust DataFrames for specific companies if necessary
filtered_list = ['Day_of_Week', 'Month', 'Quarter', 'Is_Holiday', 'Is_Month_Start', 'Is_Month_End'] + economic_indicators

# Adjust DataFrames for specific companies
special_companies = ['df_MA', 'df_MS', 'df_V', 'df_PM', 'df_GOOG']
for comp in special_companies:
    ticker = comp.split('_')[1]
    all_dfs[comp] = all_dfs[comp].loc[:, all_dfs[comp].columns.str.startswith(f'{ticker}_') | all_dfs[comp].columns.isin(filtered_list)]
    print(f"{comp} shape after adjustment: {all_dfs[comp].shape}")

    # Save adjusted DataFrame to CSV
    csv_filename = f'{comp}_adjusted.csv'
    all_dfs[comp].to_csv(csv_filename)
    print(f"Saved adjusted {comp} to {csv_filename}")

# Save column names to a text file
column_names_filename = 'combined_data_columns.txt'

with open(column_names_filename, 'w') as f:
    # Write the column names
    for column in combined_data.columns:
        f.write(f"{column}\n")

print(f"Column names saved to {column_names_filename}")

# Optional: Concatenate all company DataFrames and save to CSV
all_data_combined = pd.concat(all_dfs.values(), axis=0)
all_data_combined.to_csv('all_companies_data.csv')
print("All companies' data saved to 'all_companies_data.csv'")
