# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# %% [markdown]
# ## Constants and Parameters

# %%
sequence_length = 60  # Should match the value used in Part 1
prediction_horizon = 5
batch_size = 32

# %% [markdown]
# ## Function to Parse TFRecord Examples

# %%
def parse_tfrecord_fn(example_proto):
    feature_description = {
        'feature': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    feature = tf.io.parse_tensor(example['feature'], out_type=tf.float32)
    label = tf.io.parse_tensor(example['label'], out_type=tf.float32)
    
    # Set shapes for feature and label
    feature.set_shape([sequence_length, -1])  # -1 for number of features
    label.set_shape([prediction_horizon])
    
    return feature, label

# %% [markdown]
# ## Create tf.data.Dataset from TFRecord Files

# %%
# Get the list of TFRecord files
tfrecord_files = glob.glob('tfrecords_data/*.tfrecord')

# Create a dataset from the list of TFRecord files
raw_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP')

# Parse the serialized data in the TFRecord files
parsed_dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

# Determine the total dataset size
total_dataset_size = sum(1 for _ in parsed_dataset)
print(f"Total number of samples in dataset: {total_dataset_size}")

# %% [markdown]
# ## Split Dataset into Training and Testing Sets

# %%
# Shuffle and split the dataset
train_size = int(0.8 * total_dataset_size)
test_size = total_dataset_size - train_size

# Shuffle the entire dataset and split
parsed_dataset = parsed_dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=False)

train_dataset = parsed_dataset.take(train_size)
test_dataset = parsed_dataset.skip(train_size)

# Batch and prefetch the datasets
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

print(f"Training samples: {train_size}, Testing samples: {test_size}")

# %% [markdown]
# ## Explore the Dataset

# %%
# Get input_shape and output_length from the dataset
for features, labels in train_dataset.take(1):
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    input_shape = features.shape[1:]  # Exclude batch dimension
    output_length = labels.shape[1]
    num_features = input_shape[1]
    print(f"Input shape: {input_shape}, Output length: {output_length}, Number of features: {num_features}")
    break

# %% [markdown]
# ## Build and Train the LSTM Model

# %%
# List available GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
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
model = build_lstm_model(input_shape, output_length)

# %%
# Train the model
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=test_dataset,
    verbose=1
)

# %% [markdown]
# ## Plot Training & Validation Loss Values

# %%
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# ## Evaluate the Model

# %%
test_loss, test_mae = model.evaluate(test_dataset, verbose=1)
print(f"LSTM Model - Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# %% [markdown]
# ## Predict on Test Data

# %%
# We need to collect the actual features and labels from the test_dataset for prediction
X_test_list = []
y_test_list = []

for features, labels in test_dataset:
    X_test_list.append(features.numpy())
    y_test_list.append(labels.numpy())

# Concatenate lists to form arrays
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

# Make predictions
y_pred = model.predict(X_test)

# %% [markdown]
# ## Visualize Predictions for First Test Sample

# %%
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

# %% [markdown]
# ## Save the Trained Model

# %%
model.save('generalized_stock_lstm_model.h5')

# %% [markdown]
# ## Inference with New Data

# %% [markdown]
# ### Load Necessary Data for Inference

# %%
def load_company_data():
    all_dfs = {}
    parquet_files = glob.glob('df_*.parquet')
    for file in parquet_files:
        key = file.split('.')[0]  # e.g., 'df_AAPL'
        df = pd.read_parquet(file)
        all_dfs[key] = df
    return all_dfs

all_dfs = load_company_data()

# %% [markdown]
# ### Prepare Inference Data Function

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
        # Ensure the input sequence has the correct shape
        input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension
        return input_sequence
    else:
        raise ValueError("Insufficient data for inference (less than sequence length).")

# %% [markdown]
# ### Get LSTM Predictions for a Company

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

# %% [markdown]
# ### Load the Trained LSTM Model

# %%
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
        print(f"Processing {company_key}...")
        predictions = get_lstm_predictions_for_company(company_df, lstm_model, sequence_length=sequence_length)
        if predictions is not None:
            all_predictions[company_key] = predictions
        else:
            print(f"Predictions not available for {company_key}.")
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
                'Company': company_key.replace('df_', ''),
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
