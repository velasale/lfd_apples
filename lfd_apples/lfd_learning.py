import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


def load_data(BASE_PATH):
       
    folder = BASE_PATH    
    trials = [f for f in os.listdir(folder) if f.endswith(".csv")]

    all_arrays = []
    for trial in trials:

        file_path = os.path.join(BASE_PATH, trial)

        df = pd.read_csv(file_path)
        df.drop(columns=['timestamp_vector'], inplace=True)

        arr = df.to_numpy()
        all_arrays.append(arr)

    combined = np.vstack(all_arrays)
    
    return combined


def zscore_normalize(X_train, X_val, eps=1e-8):
    """
    Z-score normalize features using training set statistics.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + eps

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    return X_train_norm, X_val_norm, mean, std


def prepare_data(all_data, n_input_cols):
    """
    Split dataset into features and labels, train/val sets, and normalize features.

    Parameters
    ----------
    all_data : np.ndarray
        Full dataset (N, D)
    n_input_cols : int
        Number of columns corresponding to input features (rest are outputs/labels)

    Returns
    -------
    X_train_norm, Y_train, X_val_norm, Y_val
    """
    # Split features and labels
    X = all_data[:, :n_input_cols]
    Y = all_data[:, n_input_cols:]

    # Split into training and validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, shuffle=True, random_state=42
    )

    # Normalize features
    X_train_norm, X_val_norm, mean, std = zscore_normalize(X_train, X_val)

    return X_train_norm, Y_train, X_val_norm, Y_val, mean, std



def main():

    # Load data
    BASE_PATH = '/media/alejo/IL_data/04_IL_preprocessed_memory/experiment_1_(pull)/phase_1_approach/1_timesteps'
    all_data = load_data(BASE_PATH)
    
    cols = all_data.shape[1]
    output_cols = 7             # These are the action columns
    input_cols = cols - output_cols

    # Play with the number of time-stamps (t), (t-1), (t-2), ... per data input
    # short_time_memory_data = short_time_memory(all_data, input_cols)

    # Normalize data
    X_train_norm, Y_train, X_val_norm, Y_val, mean, std = prepare_data(all_data, input_cols)
  


    # Classifier
    # Initialize regressor
    rf = RandomForestRegressor(
        n_estimators=50,
        warm_start=True,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    # # --- 2. Initialize MLP ---
    # mlp = MLPRegressor(
    #     hidden_layer_sizes=(100, 100),  # two hidden layers with 50 neurons each
    #     activation='relu',
    #     solver='adam',
    #     learning_rate='adaptive',
    #     max_iter=1500,
    #     early_stopping=True,
    #     n_iter_no_change=50,
    #     random_state=42,
    #     verbose=True
    # )

    # rf = mlp
  
    
    # --- 4. Train ---
    rf.fit(X_train_norm, Y_train)

    # # --- Plot loss curve ---
    # plt.figure(figsize=(8,5))
    # plt.plot(rf.loss_curve_, label="Training Loss")

    # if hasattr(rf, "validation_scores_"):
    #     # validation_scores_ are R^2 scores per epoch
    #     val_loss = [1 - v for v in rf.validation_scores_]  # convert R^2 -> pseudo-loss
    #     plt.plot(val_loss, label="Validation (1 - R2)")

    # plt.xlabel("Epoch")
    # plt.ylabel("Loss / 1-R2")
    # plt.title("MLP Training and Validation Progress")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Review Feature Importance
    # Create DataFrame with feature importances

    df = pd.read_csv(os.path.join(BASE_PATH, 'trial_1_downsampled_aligned_data_(phase_1_approach)_(phase_1_approach)_(1_timesteps).csv'))
    df = df.iloc[:, :-7]        # simply drop action columns
    df = df.iloc[:, 1:]         # drop timevector column

    feat_df = pd.DataFrame({
        "feature": df.columns,                # column names from your data
        "importance": rf.feature_importances_
    })

    # Sort by importance descending
    feat_df = feat_df.sort_values(by="importance", ascending=False)

    # Show top 10
    top = feat_df.head(20)
    print("\n\nTop 10 Features:\n", top, '\n\n')

    
    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump(rf, f)

    # --- 4. Save normalization stats ---
    np.save("mean.npy", mean)
    np.save("std.npy", std)

    # --- 5. Predict on validation set ---
    Y_pred = rf.predict(X_val_norm)


    # Evaluate model on validation set
    # Evaluate (per output column)
    for i in range(Y_val.shape[1]):
        mse = mean_squared_error(Y_val[:, i], Y_pred[:, i])
        r2 = r2_score(Y_val[:, i], Y_pred[:, i])
        print(f"Output column {i}: MSE={mse:.4f}, R2={r2:.4f}")


if __name__ == '__main__':
    main()
