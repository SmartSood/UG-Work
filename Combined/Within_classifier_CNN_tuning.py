import os
import numpy as np
import h5py
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import keras_tuner as kt

###############################################################################
# 1. Data Loading
###############################################################################
def load_features_from_hdf5(filename):
    """
    Loads EEG features from an HDF5 file in 32x32 format.
    Returns:
      X: (n_samples, 32, 32)
      y: subject labels (e.g., "01", "02", etc.)
      runs: run labels (e.g., "Run_1", "Run_2")
      epoch_nums: epoch indices
    """
    X, y, runs, epoch_nums = [], [], [], []
    with h5py.File(filename, 'r') as h5f:
        for class_key in h5f.keys():
            subject_code = class_key.split('_', 1)[-1]
            class_group = h5f[class_key]
            for run_key in class_group.keys():
                run_group = class_group[run_key]
                for epoch_key in run_group.keys():
                    epoch_group = run_group[epoch_key]
                    feats = epoch_group['features'][()]  # shape (32,32)
                    X.append(feats)
                    y.append(subject_code)
                    runs.append(run_key)
                    ep_idx = int(epoch_key.split('_')[-1]) if '_' in epoch_key else 0
                    epoch_nums.append(ep_idx)
    return np.array(X), np.array(y), np.array(runs), np.array(epoch_nums)

###############################################################################
# 2. Hypermodel for 1D CNN using Keras Tuner
###############################################################################
def build_tunable_1d_cnn_model(hp):
    """
    Builds a 1D CNN model with hyperparameters specified via keras_tuner.
    Hyperparameters tuned include:
      - Number of filters and kernel size for the first Conv1D layer.
      - Number of filters and kernel size for the second Conv1D layer.
      - Dense layer units.
      - Dropout rate.
      - Learning rate.
    Note: Global variables 'input_shape' and 'num_classes' are defined in main().
    """
    filters_1 = hp.Int('filters_1', min_value=16, max_value=64, step=16, default=32)
    kernel_size_1 = hp.Choice('kernel_size_1', values=[3, 5, 7], default=5)
    filters_2 = hp.Int('filters_2', min_value=32, max_value=128, step=32, default=64)
    kernel_size_2 = hp.Choice('kernel_size_2', values=[3, 5], default=3)
    dense_units = hp.Int('dense_units', min_value=64, max_value=256, step=64, default=128)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
    
    # 'input_shape' and 'num_classes' are set in main() after data preparation.
    global input_shape, num_classes
    
    model = models.Sequential()
    model.add(layers.Conv1D(filters=filters_1, kernel_size=kernel_size_1, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=filters_2, kernel_size=kernel_size_2, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

###############################################################################
# 3. Main Script: Hyperparameter Tuning for 1D CNN
###############################################################################
def main():
    # Set filename and check for its existence
    filename = "all_subjects_merged_new_full_epochs.h5"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found. Please ensure the file exists.")
    
    # Load data
    X, y, runs, epoch_nums = load_features_from_hdf5(filename)
    
    # Apply PowerTransformer to the flattened data
    transformer = PowerTransformer(method='yeo-johnson')
    X_flat = X.reshape(X.shape[0], -1)
    X_tf = transformer.fit_transform(X_flat)
    # Reshape to (n_samples, 32, 28) as required by the pipeline
    X_transformed = X_tf.reshape(X.shape[0], 32, 28)
    # Save the transformer for future use
    joblib.dump(transformer, "power_transformer.pkl")
    
    # Split data into training and testing sets based on run labels
    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Insufficient data for Run_1 or Run_2. Please check your run labels.")
    
    X_train = X_transformed[train_idx]
    X_test = X_transformed[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    # Flatten the images for CNN: each sample becomes a 1D sequence
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Define global input shape and number of classes (needed in build_tunable_1d_cnn_model)
    global input_shape, num_classes
    flattened_length = X_train_flat.shape[1]
    input_shape = (flattened_length, 1)
    
    # Reshape data to (n_samples, flattened_length, 1) for the CNN
    X_train_cnn = X_train_flat.reshape(X_train_flat.shape[0], flattened_length, 1)
    X_test_cnn = X_test_flat.reshape(X_test_flat.shape[0], flattened_length, 1)
    
    # Encode labels and convert to one-hot (categorical) format
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(np.unique(y_train_enc))
    y_train_cat = utils.to_categorical(y_train_enc, num_classes=num_classes)
    y_test_cat = utils.to_categorical(y_test_enc, num_classes=num_classes)
    
    # Early stopping callback to prevent overfitting
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Initialize Keras Tuner with RandomSearch
    tuner = kt.RandomSearch(
        build_tunable_1d_cnn_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='kt_1d_cnn_tuning',
        project_name='1d_cnn'
    )
    
    # Run the hyperparameter search
    tuner.search(X_train_cnn, y_train_cat, epochs=20, validation_split=0.1, callbacks=[early_stop], verbose=1)
    
    # Retrieve the best hyperparameters and print them
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("The optimal hyperparameters are:")
    print(f"  Filters in first Conv1D layer: {best_hp.get('filters_1')}")
    print(f"  Kernel size in first Conv1D layer: {best_hp.get('kernel_size_1')}")
    print(f"  Filters in second Conv1D layer: {best_hp.get('filters_2')}")
    print(f"  Kernel size in second Conv1D layer: {best_hp.get('kernel_size_2')}")
    print(f"  Dense layer units: {best_hp.get('dense_units')}")
    print(f"  Dropout rate: {best_hp.get('dropout_rate')}")
    print(f"  Learning rate: {best_hp.get('learning_rate')}")
    
    # Build and train the best model with the optimal hyperparameters
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(X_train_cnn, y_train_cat, epochs=20, validation_split=0.1, callbacks=[early_stop], verbose=1)
    
    # Evaluate the best model on the test set
    test_loss, test_acc = best_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Generate predictions and compute additional evaluation metrics
    predictions_prob = best_model.predict(X_test_cnn)
    predictions_enc = np.argmax(predictions_prob, axis=1)
    predictions = le.inverse_transform(predictions_enc)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, predictions)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Save the best model and the label encoder for future use
    best_model.save("best_1d_cnn_model.h5")
    joblib.dump(le, "label_encoder.pkl")

if __name__ == "__main__":
    main()
