# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, MultiHeadAttention, LayerNormalization, Add
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import h5py  # For reading HDF5 files

# %% [markdown]
# ## Load Preprocessed Data from HDF5

# %%
# Open the HDF5 file and load the datasets
with h5py.File('sequence_data.h5', 'r') as hf:
    X = hf['X'][:]
    y = hf['y'][:]

print(f"Loaded X shape: {X.shape}")
print(f"Loaded y shape: {y.shape}")

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data Shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing Data Shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

# %% [markdown]
# ## Add Positional Encoding Function

# %%
def positional_encoding(sequence_length, d_model):
    import numpy as np
    position = np.arange(sequence_length)[:, np.newaxis]  # shape (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((sequence_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, ...]
    return tf.cast(pe, dtype=tf.float32)

# %% [markdown]
# ## Build and Train the Galformer Model (Transformer-based Model)

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
# Define the Galformer (Transformer) model
def build_galformer_model(input_shape, output_length):
    inputs = Input(shape=input_shape)  # input_shape = (sequence_length, num_features)
    
    # Create positional encodings
    pe = positional_encoding(input_shape[0], input_shape[1])
    
    # Add positional encoding to inputs
    x = Add()([inputs, pe])
    
    # Transformer Encoder Layer
    attn_output = MultiHeadAttention(num_heads=4, key_dim=input_shape[1])(x, x)
    attn_output = Dropout(0.1)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(attn_output + x)

    # Feed Forward Network
    ffn_output = Dense(128, activation='relu')(out1)
    ffn_output = Dense(input_shape[1])(ffn_output)
    ffn_output = Dropout(0.1)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(ffn_output + out1)

    # Flatten and Output Layer
    x = Flatten()(out2)
    outputs = Dense(output_length, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Initialize the Galformer model
input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
output_length = y_train.shape[1]  # prediction_horizon
galformer_model = build_galformer_model(input_shape, output_length)

# %%
# Train the Galformer model
history_galformer = galformer_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# %%
# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history_galformer.history['loss'], label='Train Loss')
plt.plot(history_galformer.history['val_loss'], label='Validation Loss')
plt.title('Galformer Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# Evaluate the Galformer model
test_loss_galformer, test_mae_galformer = galformer_model.evaluate(X_test, y_test, verbose=1)
print(f"Galformer Model - Test Loss: {test_loss_galformer:.4f}, Test MAE: {test_mae_galformer:.4f}")

# Predict on test data
y_pred_galformer = galformer_model.predict(X_test)

# Visualize predictions for the first test sample
plt.figure(figsize=(10, 6))

# Plot Actual Prices for the first test sample
plt.plot(range(1, output_length + 1), y_test[0], label="Actual Prices", marker='o')

# Plot Predicted Prices for the first test sample (Galformer)
plt.plot(range(1, output_length + 1), y_pred_galformer[0], label="Predicted Prices (Galformer)", marker='x')

plt.title("Actual vs Predicted Prices (First Test Sample - Galformer)")
plt.xlabel("Days Ahead")
plt.ylabel("Price")
plt.legend()
plt.show()

# %%
# Save the Galformer model
galformer_model.save('generalized_stock_galformer_model.h5')

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
def get_galformer_predictions_for_company(company_df, galformer_model, sequence_length=60):
    """
    Get Galformer predictions for a single company using the trained model.
    Args:
        company_df (DataFrame): DataFrame of the company.
        galformer_model: Trained Galformer model.
        sequence_length (int): Number of past days to consider as input.

    Returns:
        numpy array: Predicted prices.
    """
    try:
        # Prepare data for inference
        input_data = prepare_inference_data(company_df, sequence_length=sequence_length)
        
        # Add positional encoding to inference data
        pe = positional_encoding(input_shape[0], input_shape[1])
        input_data_with_pe = input_data + pe.numpy()
        
        # Make predictions with Galformer
        pred_galformer = galformer_model.predict(input_data_with_pe)
        return pred_galformer.flatten()
    except ValueError as e:
        print(f"Skipping due to error: {e}")
        return None

# %%
# Load the trained Galformer model
galformer_model = load_model('generalized_stock_galformer_model.h5')

# %% [markdown]
# ### Example: Making Predictions for a Specific Company

# %%
# Choose a company
company_key = 'df_AAPL'  # Example company
if company_key in all_dfs:
    company_df = all_dfs[company_key]
    predictions = get_galformer_predictions_for_company(company_df, galformer_model, sequence_length=60)
    
    if predictions is not None:
        # Visualize predictions
        plt.figure(figsize=(10, 6))
        
        # Plot Galformer Predictions
        plt.plot(range(1, len(predictions) + 1), predictions, marker='x', label='Predicted Prices (Galformer)')
        
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
def get_galformer_predictions_for_all_companies(all_dfs, galformer_model, sequence_length=60):
    all_predictions = {}
    for company_key, company_df in all_dfs.items():
        predictions = get_galformer_predictions_for_company(company_df, galformer_model, sequence_length=60)
        if predictions is not None:
            all_predictions[company_key] = predictions
            print(f"Predictions for {company_key}:")
            print("Galformer Predictions:", predictions)
            print("-----------------------------")
    return all_predictions

# %%
# Get predictions for all companies
all_company_predictions = get_galformer_predictions_for_all_companies(all_dfs, galformer_model, sequence_length=60)

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
predictions_df.to_csv('galformer_stock_price_predictions.csv', index=False)
print("Galformer Predictions have been saved to 'galformer_stock_price_predictions.csv'.")
