# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Constants and Parameters
sequence_length = 60  # Should match the value used in data preparation
prediction_horizon = 5  # Number of days ahead to predict
epochs = 100
batch_size = 512  # Adjust based on your system's capabilities
print('Batch size:', batch_size)

# Load Prepared CSV Data
csv_files = glob.glob('df_*.csv')  # Adjust the path if necessary
print(f"Found {len(csv_files)} company CSV files.")

# Initialize lists to collect data
X_list = []
y_list = []
ticker_list = []

# For saving feature columns
feature_columns_saved = False
common_feature_columns = None  # List to store common features

# Loop through each company's CSV file
for csv_file in csv_files:
    print(f"Processing {csv_file}...")
    # Read the CSV file
    df = pd.read_csv(csv_file, parse_dates=['Date'], index_col='Date')

    # Ensure DataFrame is sorted by date
    df.sort_index(inplace=True)

    # Extract the ticker symbol from the filename
    ticker = csv_file.split('_')[1].replace('.csv', '').replace('adjusted', '').strip()

    # Remove ticker prefixes from column names
    df.columns = [col.replace(f'{ticker}_', '') if col.startswith(f'{ticker}_') else col for col in df.columns]

    # Add Ticker as a column
    df['Ticker'] = ticker

    # Select relevant input features (exclude target columns)
    input_features = df.filter(regex="^(?!.*target).*")

    # Select target columns (without ticker prefix)
    target_columns = df.filter(regex="target").columns.tolist()

    # Save the feature columns from the first company
    if not feature_columns_saved:
        # One-Hot Encode the 'Ticker' column
        input_features = pd.get_dummies(input_features, columns=['Ticker'])
        common_feature_columns = input_features.columns.tolist()
        with open('model_feature_columns.txt', 'w') as f:
            for col in common_feature_columns:
                f.write(f"{col}\n")
        feature_columns_saved = True
        print("Feature columns saved to 'model_feature_columns.txt'")
    else:
        # One-Hot Encode the 'Ticker' column
        input_features = pd.get_dummies(input_features, columns=['Ticker'])
        # Align input_features to common_feature_columns
        input_features = input_features.reindex(columns=common_feature_columns, fill_value=0)

    # Check if all features are present
    if input_features.isnull().values.any():
        print(f"NaN values found in input features for {csv_file} after reindexing.")
        input_features = input_features.fillna(0)

    # Ensure target columns are consistent with prediction_horizon
    if len(target_columns) < prediction_horizon:
        print(f"Not enough target columns in {csv_file}. Expected {prediction_horizon}, found {len(target_columns)}. Skipping this company.")
        continue
    else:
        # Trim target columns if necessary
        target_columns = target_columns[:prediction_horizon]

    # Prepare sequences for this company
    def create_sequences(input_data, target_data, seq_length, pred_horizon):
        X = []
        y = []
        for i in range(len(input_data) - seq_length - pred_horizon + 1):
            X.append(input_data.iloc[i:i+seq_length].values)
            y.append(target_data.iloc[i+seq_length:i+seq_length+pred_horizon].values.flatten())
        return np.array(X), np.array(y)

    X_company, y_company = create_sequences(
        input_data=input_features,
        target_data=df[target_columns],
        seq_length=sequence_length,
        pred_horizon=prediction_horizon
    )

    # Check if sequences are created
    if X_company.size == 0 or y_company.size == 0:
        print(f"No sequences created for {csv_file}. Skipping this company.")
        continue

    # Add to the lists
    X_list.append(X_company)
    y_list.append(y_company)

# Check if any data was collected
if not X_list or not y_list:
    print("No data collected. Exiting.")
    exit()

# Concatenate data from all companies
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

print(f"Final shape of X: {X.shape}")
print(f"Final shape of y: {y.shape}")

# Convert Data Types and Normalize
X = X.astype(np.float32)
y = y.astype(np.float32)

# Normalize features and labels per sample
label_mean_std = []

for i in range(X.shape[0]):
    # Normalize features
    feature_mean = np.mean(X[i], axis=0, keepdims=True)
    feature_std = np.std(X[i], axis=0, keepdims=True) + 1e-6
    X[i] = (X[i] - feature_mean) / feature_std

    # Normalize labels
    label_mean = np.mean(y[i])
    label_std = np.std(y[i]) + 1e-6
    y[i] = (y[i] - label_mean) / label_std

    label_mean_std.append([label_mean, label_std])

label_mean_std = np.array(label_mean_std, dtype=np.float32)

print(f"Data types after conversion and normalization: X - {X.dtype}, y - {y.dtype}")

# Save Label Mean and Std for Inverse Transformation
np.save('label_mean_std.npy', label_mean_std)
print("Label mean and std saved to 'label_mean_std.npy'.")

# Create TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Shuffle and split the dataset
total_dataset_size = X.shape[0]
train_size = int(0.8 * total_dataset_size)
test_size = total_dataset_size - train_size

# Shuffle the entire dataset
dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=False)

# Split into training and testing datasets
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Batch and prefetch the datasets
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

print(f"Total samples: {total_dataset_size}, Training samples: {train_size}, Testing samples: {test_size}")

# Define Positional Encoding Function
def positional_encoding(sequence_length, d_model):
    position = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = position * angle_rates
    sin_terms = tf.sin(angle_rads[:, 0::2])
    cos_terms = tf.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sin_terms, cos_terms], axis=-1)
    return pos_encoding  # Shape: (sequence_length, d_model)

# Build and Compile the Galformer Model
num_features = X.shape[2]
input_shape = (sequence_length, num_features)  # (sequence_length, num_features)
output_length = prediction_horizon  # prediction_horizon

def build_galformer_model(input_shape, output_length, num_layers=3, dff=512, num_heads=8):
    inputs = Input(shape=input_shape)  # input_shape = (sequence_length, num_features)

    # Create positional encodings
    pe = positional_encoding(input_shape[0], input_shape[1])
    pe = tf.expand_dims(pe, axis=0)  # Shape: (1, sequence_length, num_features)

    # Add positional encoding to inputs
    x = inputs + pe  # Broadcasting, pe shape is (1, sequence_length, num_features)

    # Stack multiple Transformer Encoder Layers
    for _ in range(num_layers):
        # Multi-Head Attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1], dropout=0.1)(x, x)
        attn_output = Dropout(0.2)(attn_output)  # Increased dropout
        out1 = LayerNormalization(epsilon=1e-6)(attn_output + x)

        # Feed Forward Network with L2 Regularization
        ffn_output = Dense(dff, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(out1)
        ffn_output = Dense(input_shape[1], kernel_regularizer=regularizers.l2(1e-4))(ffn_output)
        ffn_output = Dropout(0.2)(ffn_output)  # Increased dropout
        x = LayerNormalization(epsilon=1e-6)(ffn_output + out1)

    # Flatten and Output Layer
    x = Flatten()(x)
    outputs = Dense(output_length, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    # Define learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-4,  # Reduced learning rate
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

galformer_model = build_galformer_model(input_shape, output_length)
galformer_model.summary()

# Training Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)

callbacks = [early_stopping, lr_scheduler]

# Train the Galformer Model
history_galformer = galformer_model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)

# Plot Training & Validation Loss Values
plt.figure(figsize=(10, 6))
plt.plot(history_galformer.history['loss'], label='Train Loss')
plt.plot(history_galformer.history['val_loss'], label='Validation Loss')
plt.title('Galformer Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('train_validation_loss.png')
plt.close()  # Close the plot to free memory

# Evaluate the Galformer Model
test_loss_galformer, test_mae_galformer = galformer_model.evaluate(test_dataset, verbose=1)
print(f"Galformer Model - Test Loss: {test_loss_galformer:.4f}, Test MAE: {test_mae_galformer:.4f}")

# Predict on Test Data
X_test = []
y_test = []
for features_batch, labels_batch in test_dataset:
    X_test.append(features_batch.numpy())
    y_test.append(labels_batch.numpy())

X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

label_mean_std_test = label_mean_std[train_size:]

X_test_with_pe = X_test + positional_encoding(sequence_length, num_features).numpy()

y_pred_galformer = galformer_model.predict(X_test_with_pe)

# Denormalize predictions and actual labels
label_mean = label_mean_std_test[:, 0]  # Shape: (num_samples,)
label_std = label_mean_std_test[:, 1]   # Shape: (num_samples,)

# Reshape label_mean and label_std to (num_samples, 1) to match the predictions
label_mean = label_mean.reshape(-1, 1)
label_std = label_std.reshape(-1, 1)

y_pred_galformer_denorm = y_pred_galformer * label_std + label_mean
y_test_denorm = y_test * label_std + label_mean

# Visualize predictions for the first test sample
plt.figure(figsize=(10, 6))
plt.plot(range(1, output_length + 1), y_test_denorm[0], label="Actual Prices", marker='o')
plt.plot(range(1, output_length + 1), y_pred_galformer_denorm[0], label="Predicted Prices (Galformer)", marker='x')
plt.title("Actual vs Predicted Prices (First Test Sample - Galformer)")
plt.xlabel("Days Ahead")
plt.ylabel("Price")
plt.legend()
plt.savefig('actual_vs_predicted_prices.png')  # Save the plot
plt.close()  # Close the plot to free memory

# Save the Galformer Model
galformer_model.save('enhanced_stock_galformer_model.keras')
print("Galformer Model has been saved to 'enhanced_stock_galformer_model.keras'.")
