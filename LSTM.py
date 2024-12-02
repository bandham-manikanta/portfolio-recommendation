# %% [markdown]
# # Enhanced LSTM Model Training and Inference

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# %% [markdown]
# ## Constants and Parameters

# %%
sequence_length = 60  # Sequence length matching the data preparation
prediction_horizon = 25  # Matching Galformer prediction horizon
batch_size = 512  # Matching Galformer batch size
epochs = 1  # Increased epochs for better training
print('Batch size:', batch_size)

# %% [markdown]
# ## Function to Parse and Normalize TFRecord Examples

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
    feature.set_shape([sequence_length, None])  # None for number of features
    label.set_shape([prediction_horizon])
    
    # Normalize features (z-score standardization)
    feature_mean = tf.reduce_mean(feature, axis=0, keepdims=True)
    feature_std = tf.math.reduce_std(feature, axis=0, keepdims=True) + 1e-6
    feature = (feature - feature_mean) / feature_std
    
    # Normalize labels (z-score standardization)
    label_mean = tf.reduce_mean(label)
    label_std = tf.math.reduce_std(label) + 1e-6
    label = (label - label_mean) / label_std
    
    # Store label mean and std for inverse transformation
    label_info = tf.stack([label_mean, label_std])
    
    return feature, (label, label_info)

# %% [markdown]
# ## Create `tf.data.Dataset` from TFRecord Files

# %%
# Get the list of TFRecord files
tfrecord_files = glob.glob('tfrecords_data/*.tfrecord')
print(f"Found {len(tfrecord_files)} TFRecord files.")

# Create a dataset from the list of TFRecord files
raw_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP')

# Parse the serialized data in the TFRecord files
parsed_dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

# Determine the total dataset size
total_dataset_size = sum(1 for _ in parsed_dataset)
print(f"Total number of samples in dataset: {total_dataset_size}")

# Reset the parsed_dataset iterator after counting
parsed_dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

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
# ## Prepare the Dataset for Training

# %%
# Function to extract labels (normalized labels are in y[0])
def strip_label_info(x, y):
    return x, y[0]

train_dataset_for_training = train_dataset.map(strip_label_info)
test_dataset_for_training = test_dataset.map(strip_label_info)

# Get input_shape and output_length from the dataset
for features, labels in train_dataset_for_training.take(1):
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    input_shape = features.shape[1:]  # Exclude batch dimension
    output_length = labels.shape[1]
    num_features = input_shape[1]
    print(f"Input shape: {input_shape}, Output length: {output_length}, Number of features: {num_features}")
    break

# %% [markdown]
# ## Build and Train the Enhanced LSTM Model

# %%
# Define the enhanced LSTM model
def build_enhanced_lstm_model(input_shape, output_length):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), input_shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Bidirectional(LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.3),
        Dense(output_length, activation='linear')
    ])
    
    # Define learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Initialize the model
model = build_enhanced_lstm_model(input_shape, output_length)

# %%
# Early Stopping and Learning Rate Scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)

callbacks = [early_stopping, lr_scheduler]

# Train the model
history = model.fit(
    train_dataset_for_training,
    epochs=epochs,
    validation_data=test_dataset_for_training,
    callbacks=callbacks,
    verbose=1
)

# %% [markdown]
# ## Plot Training & Validation Loss Values

# %%
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Enhanced LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('enhanced_lstm_train_validation_loss.png')
plt.close()

# %% [markdown]
# ## Evaluate the Model

# %%
test_loss, test_mae = model.evaluate(test_dataset_for_training, verbose=1)
print(f"Enhanced LSTM Model - Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# %% [markdown]
# ## Predict on Test Data and Denormalize

# %%
X_test_list = []
y_test_list = []
label_info_list = []

for features, (labels, label_info) in test_dataset:
    X_test_list.append(features.numpy())
    y_test_list.append(labels.numpy())
    label_info_list.append(label_info.numpy())

# Concatenate lists to form arrays
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)
label_info = np.concatenate(label_info_list, axis=0)  # Shape: (num_samples, 2)

# Make predictions
y_pred = model.predict(X_test)

# Denormalize predictions and actual labels
label_mean = label_info[:, 0]  # Shape: (num_samples,)
label_std = label_info[:, 1]   # Shape: (num_samples,)

# Reshape label_mean and label_std to (num_samples, 1) to match the predictions
label_mean = label_mean.reshape(-1, 1)
label_std = label_std.reshape(-1, 1)

# Denormalize predictions and actual labels
y_pred_denorm = y_pred * label_std + label_mean
y_test_denorm = y_test * label_std + label_mean

# %% [markdown]
# ## Visualize Predictions for First Test Sample

# %%
plt.figure(figsize=(10, 6))

# Plot Actual Prices for the first test sample
plt.plot(range(1, output_length + 1), y_test_denorm[0], label="Actual Prices", marker='o')

# Plot Predicted Prices for the first test sample
plt.plot(range(1, output_length + 1), y_pred_denorm[0], label="Predicted Prices (Enhanced LSTM)", marker='x')

plt.title("Actual vs Predicted Prices (First Test Sample - Enhanced LSTM)")
plt.xlabel("Days Ahead")
plt.ylabel("Price")
plt.legend()
plt.savefig('enhanced_lstm_actual_vs_predicted_prices.png')
plt.close()

# %% [markdown]
# ## Save the Trained Model

# %%
model.save('enhanced_stock_lstm_model.keras')
print("Enhanced LSTM Model has been saved to 'enhanced_stock_lstm_model.keras'.")

# %% [markdown]
# ## Inference with New Data

# %% [markdown]
# ### Load Necessary Data for Inference

# %%
def load_company_data():
    all_dfs = {}
    parquet_files = glob.glob('df_*.parquet')
    for file in parquet_files:
        key = os.path.splitext(os.path.basename(file))[0]  # e.g., 'df_AAPL'
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
        # Normalize features using the same method as during training
        feature_mean = np.mean(input_sequence, axis=0, keepdims=True)
        feature_std = np.std(input_sequence, axis=0, keepdims=True) + 1e-6
        input_sequence = (input_sequence - feature_mean) / feature_std
        # Ensure the input sequence has the correct shape
        input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension
        return input_sequence.astype(np.float32), feature_mean, feature_std
    else:
        print(f"Insufficient data for inference (less than sequence length) for company.")
        return None, None, None

# %% [markdown]
# ### Get Enhanced LSTM Predictions for a Company

# %%
def get_enhanced_lstm_predictions_for_company(company_df, lstm_model, sequence_length=60):
    """
    Get Enhanced LSTM predictions for a single company using the trained model.
    Args:
        company_df (DataFrame): DataFrame of the company.
        lstm_model: Trained Enhanced LSTM model.
        sequence_length (int): Number of past days to consider as input.

    Returns:
        numpy array: Predicted prices (standardized).
    """
    try:
        # Prepare data for inference
        input_data, _, _ = prepare_inference_data(company_df, sequence_length=sequence_length)
        if input_data is None:
            return None
        
        # Make predictions with Enhanced LSTM
        pred_lstm = lstm_model.predict(input_data)
        # Note: Since labels were normalized per sample during training, and during inference we don't have label mean and std,
        # the predictions are in standardized form and cannot be directly denormalized.
        return pred_lstm.flatten()
    except ValueError as e:
        print(f"Skipping due to error: {e}")
        return None

# %% [markdown]
# ### Load the Trained Enhanced LSTM Model

# %%
lstm_model = tf.keras.models.load_model('enhanced_stock_lstm_model.keras')

# %% [markdown]
# ### Making Predictions for All Companies

# %%
# Function to get predictions for all companies
def get_enhanced_lstm_predictions_for_all_companies(all_dfs, lstm_model, sequence_length=60):
    all_predictions = {}
    for company_key, company_df in all_dfs.items():
        print(f"Processing {company_key}...")
        predictions = get_enhanced_lstm_predictions_for_company(company_df, lstm_model, sequence_length=sequence_length)
        if predictions is not None:
            all_predictions[company_key] = predictions
        else:
            print(f"Predictions not available for {company_key}.")
    return all_predictions

# %%
# Get predictions for all companies
all_company_predictions = get_enhanced_lstm_predictions_for_all_companies(all_dfs, lstm_model, sequence_length=60)

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
                'Predicted_Price_Standardized': value  # Note: Standardized predictions
            })
    return pd.DataFrame(records)

predictions_df = predictions_to_dataframe(all_company_predictions)
predictions_df.head()

# %%
# Save the predictions DataFrame to a CSV file
predictions_df.to_csv('enhanced_lstm_stock_price_predictions.csv', index=False)
print("Enhanced LSTM Predictions have been saved to 'enhanced_lstm_stock_price_predictions.csv'.")
