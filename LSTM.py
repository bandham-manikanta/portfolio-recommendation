# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob

# %% [markdown]
# ## Load Preprocessed Data

# %%
# Load the preprocessed data
X = np.load('X_sequence_data.npy')
y = np.load('y_sequence_data.npy')

print(f"Loaded X shape: {X.shape}")
print(f"Loaded y shape: {y.shape}")

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data Shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing Data Shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

# %% [markdown]
# ## Build and Train the LSTM Model

# %%
# List available GPUs
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
# Set memory growth for GPUs
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# %%
# Define the LSTM model
def build_lstm_model(input_shape, output_length):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(output_length, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Initialize the model
input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
output_length = y_train.shape[1]  # prediction_horizon (number of days to predict)
model = build_lstm_model(input_shape, output_length)

# %%
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# %%
# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"LSTM Model - Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Predict on test data
y_pred = model.predict(X_test)

# Visualize predictions for the first test sample
plt.figure(figsize=(10, 6))

# Plot Actual Prices for the first test sample
plt.plot(range(1, output_length + 1), y_test[0], label="Actual Prices", marker='o')

# Plot Predicted Prices for the first test sample
plt.plot(range(1, output_length + 1), y_pred[0], label="Predicted Prices (LSTM)", marker='x')

plt.title("Actual vs Predicted Prices (First Test Sample - LSTM)")
plt.xlabel("Days Ahead")
plt.ylabel("Price")
plt.legend()
plt.show()

# %%
# Save the trained model
model.save('generalized_stock_lstm_model.h5')

# %% [markdown]
# ## Inference with New Data

# %%
# Load necessary data for inference
def load_company_data():
    all_dfs = {}
    parquet_files = glob.glob('df_*.parquet')
    for file in parquet_files:
        key = file.split('.')[0]  # e.g., 'df_AAPL'
        df = pd.read_parquet(file)
        all_dfs[key] = df
    return all_dfs

all_dfs = load_company_data()

# %%
def prepare_inference_data(company_df, sequence_length=60):
    """
    Prepare input data for inference for a single company.
    Args:
        company_df (DataFrame): The DataFrame for a specific company.
        sequence_length (int): The number of past days to consider as input.

    Returns:
        numpy array: The input data ready for prediction.
    """
    # Ensure data is sorted by date
    company_df = company_df.sort_index()

    # Select relevant input features (exclude targets)
    input_features = company_df.filter(regex="^(?!.*target).*").values

    # Take the last `sequence_length` days as input for prediction
    if len(input_features) >= sequence_length:
        input_sequence = input_features[-sequence_length:]
        return np.expand_dims(input_sequence, axis=0)  # Add batch dimension
    else:
        raise ValueError("Insufficient data for inference (less than sequence length).")

# %%
def get_lstm_predictions_for_company(company_df, lstm_model, sequence_length=60):
    """
    Get LSTM predictions for a single company using the trained model.
    Args:
        company_df (DataFrame): DataFrame of the company.
        lstm_model: Trained LSTM model.
        sequence_length (int): Number of past days to consider as input.

    Returns:
        numpy array: Predicted prices.
    """
    try:
        # Prepare data for inference
        input_data = prepare_inference_data(company_df, sequence_length=sequence_length)
        
        # Make predictions with LSTM
        pred_lstm = lstm_model.predict(input_data)
        return pred_lstm.flatten()
    except ValueError as e:
        print(f"Skipping due to error: {e}")
        return None

# %%
# Load the trained LSTM model
lstm_model = load_model('generalized_stock_lstm_model.h5')

# %% [markdown]
# ### Example: Making Predictions for a Specific Company

# %%
# Choose a company
company_key = 'df_AAPL'  # Example company
if company_key in all_dfs:
    company_df = all_dfs[company_key]
    predictions = get_lstm_predictions_for_company(company_df, lstm_model, sequence_length=60)
    
    if predictions is not None:
        # Visualize predictions
        plt.figure(figsize=(10, 6))
        
        # Plot LSTM Predictions
        plt.plot(range(1, len(predictions) + 1), predictions, marker='o', label='Predicted Prices (LSTM)')
        
        plt.title(f"Predicted Prices for {company_key} (Next {len(predictions)} Days)")
        plt.xlabel("Days Ahead")
        plt.ylabel("Price")
        plt.legend()
        plt.show()
else:
    print(f"No data available for {company_key}.")

# %% [markdown]
# ### Making Predictions for All Companies

# %%
# Function to get predictions for all companies
def get_lstm_predictions_for_all_companies(all_dfs, lstm_model, sequence_length=60):
    all_predictions = {}
    for company_key, company_df in all_dfs.items():
        predictions = get_lstm_predictions_for_company(company_df, lstm_model, sequence_length=60)
        if predictions is not None:
            all_predictions[company_key] = predictions
            print(f"Predictions for {company_key}:")
            print("LSTM Predictions:", predictions)
            print("-----------------------------")
    return all_predictions

# %%
# Get predictions for all companies
all_company_predictions = get_lstm_predictions_for_all_companies(all_dfs, lstm_model, sequence_length=60)

# %% [markdown]
# ### Saving Predictions

# %%
# Convert predictions to DataFrame for further analysis or saving
def predictions_to_dataframe(predictions_dict):
    records = []
    for company_key, pred_values in predictions_dict.items():
        for day_ahead, value in enumerate(pred_values, start=1):
            records.append({
                'Company': company_key,
                'Day_Ahead': day_ahead,
                'Predicted_Price': value
            })
    return pd.DataFrame(records)

predictions_df = predictions_to_dataframe(all_company_predictions)
predictions_df.head()

# %%
# Save the predictions DataFrame to a CSV file
predictions_df.to_csv('lstm_stock_price_predictions.csv', index=False)
print("LSTM Predictions have been saved to 'lstm_stock_price_predictions.csv'.")
