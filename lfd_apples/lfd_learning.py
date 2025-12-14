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
import joblib
import yaml

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(BASE_PATH):
       
    folder = BASE_PATH    
    trials = [f for f in os.listdir(folder) if f.endswith(".csv")]

    all_arrays = []
    filepaths = []
    for trial in trials:

        file_path = os.path.join(BASE_PATH, trial)

        df = pd.read_csv(file_path)
        df.drop(columns=['timestamp_vector'], inplace=True)
       
        arr = df.to_numpy()
        all_arrays.append(arr)

        filepaths.append(file_path)

    combined = np.vstack(all_arrays)
    
    return combined, filepaths


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
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=True, random_state=42
    )

    # Normalize features
    X_train_norm, X_test_norm, mean, std = zscore_normalize(X_train, X_test)

    return X_train_norm, Y_train, X_test_norm, Y_test, mean, std


def prepare_data_approach2(train_trials_list, test_trials_list, n_input_cols):
    """
    Docstring for prepare_data_approach2
    
    :param train_trials_list: Description
    :param test_trials_list: Description
    :param n_input_cols: Description
    """
   
   # Prepare Training Set
    train_arrays = []
    for trial in train_trials_list:      

        df = pd.read_csv(trial)
        df = df.apply(pd.to_numeric, errors='coerce')

        df.drop(columns=['timestamp_vector'], inplace=True)
       
        arr = df.to_numpy(dtype=np.float64)
        train_arrays.append(arr)

    train_combined = np.vstack(train_arrays)
    X_train = train_combined[:, :n_input_cols]
    Y_train = train_combined[:, n_input_cols:]

    # Clip linear velocities (columns 0,1,2) and angular velocities (columns 3,4,5)
    Y_train_clipped = Y_train.copy()
    linear_max = 0.5      # m/s
    angular_max = 0.5      # rad/s
    Y_train_clipped[:, 0:3] = np.clip(Y_train_clipped[:, 0:3], -linear_max, linear_max)    
    Y_train_clipped[:, 3:6] = np.clip(Y_train_clipped[:, 3:6], -angular_max, angular_max)

    # Prepare Testing Set
    test_arrays = []
    for trial in test_trials_list:      

        df = pd.read_csv(trial)
        df.drop(columns=['timestamp_vector'], inplace=True)
       
        arr = df.to_numpy()
        test_arrays.append(arr)

    test_combined = np.vstack(test_arrays)
    X_test = test_combined[:, :n_input_cols]
    Y_test = test_combined[:, n_input_cols:]

    # Clip linear velocities (columns 0,1,2) and angular velocities (columns 3,4,5)
    Y_test_clipped = Y_test.copy()
    Y_test_clipped[:, 0:3] = np.clip(Y_test_clipped[:, 0:3], -linear_max, linear_max)    
    Y_test_clipped[:, 3:6] = np.clip(Y_test_clipped[:, 3:6], -angular_max, angular_max)

    # Normalize features
    X_train_norm, X_test_norm, X_mean, X_std = zscore_normalize(X_train, X_test)
    Y_train_norm, Y_test_norm, Y_mean, Y_std = zscore_normalize(Y_train_clipped, Y_test_clipped)    

    nan_rows = np.isnan(X_train_norm).any(axis=1)
    print(f"Rows with NaNs in X_train_nrom: {np.where(nan_rows)[0]}")
    nan_rows = np.isnan(Y_train).any(axis=1)
    print(f"Rows with NaNs in Y_train: {np.where(nan_rows)[0]}")

    return X_train_norm, Y_train_norm, X_test_norm, Y_test_clipped, X_mean, X_std, Y_mean, Y_std


def learn(regressor='mlp', phase='phase_1_approach', time_steps='2_timesteps'):
    """
    Docstring for learn
    
    :param regressor: 'rf' or 'mlp'  Random Forest or Multi-Layer Perceptron
    :param phase: 'phase_1_approach', 'phase_2_contact', 'phase_3_pick'
    :param time_steps: Description
    """
    
    # Load Data
    # BASE_DIRECTORY = '/media/alejo/IL_data/04_IL_preprocessed_(memory)'
    BASE_SOURCE_PATH = '/home/alejo/Documents/DATA'
    BASE_DIRECTORY = os.path.join(BASE_SOURCE_PATH, '04_IL_preprocessed_(memory)')
    experiment = 'experiment_1_(pull)'    
   
    suffix = '_' + experiment + '_' + phase + '_' + time_steps       
    BASE_PATH = os.path.join(BASE_DIRECTORY, experiment, phase, time_steps)

    DESTINATION_DIRECTORY = os.path.join(BASE_SOURCE_PATH, '05_IL_learning')
    DESTINATION_PATH = os.path.join(DESTINATION_DIRECTORY, experiment, phase, time_steps)
    os.makedirs(DESTINATION_PATH, exist_ok=True)

    # Split data into Training trials and Test trials
    all_data, all_trials_paths = load_data(BASE_PATH)    
    cols = all_data.shape[1]

    # Check size of actions
    data_columns_path = config_path = Path(__file__).parent / "config" / "lfd_data_columns.yaml"
    with open(data_columns_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    n_output_cols = len(cfg['action_cols'])                         # These are the action columns
    n_input_cols = cols - n_output_cols  

    # Approach 1: Split data row-wise
    # Normalize data
    # X_train_norm, Y_train, X_test_norm, Y_test, mean, std = prepare_data(all_data, input_cols)
  

    # Approach 2: Split data trial-wise
    train_trials, test_trials = train_test_split(
        all_trials_paths, test_size=0.20, shuffle=True, random_state=42
    )
    X_train_norm, Y_train_norm, X_test_norm, Y_test, x_mean, x_std, y_mean, y_std = prepare_data_approach2(train_trials, test_trials, n_input_cols)

    # Save Y_training set as csv for analysis
    Y_train_df = pd.DataFrame(Y_train_norm)
    Y_train_df.to_csv(os.path.join(DESTINATION_PATH, 'Y_train_normalized.csv'), index=False)

    # plot distribution of Y_train_norm
    plt.figure(figsize=(10,6))
    for i in range(Y_train_norm.shape[1]):
        plt.subplot(Y_train_norm.shape[1], 1, i+1)
        plt.hist(Y_train_norm[:, i], bins=50, alpha=0.7)
        plt.title(f'Distribution of Y_train_norm Column {i}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(DESTINATION_PATH, 'Y_train_normalized_distribution.png'), dpi=300)
    plt.close()

    # Linear Regression
    if regressor == 'rf':

        # ==================== RANDOM FOREST REGRESSOR ======================
            
        # --- Initialize regressor ---
        regressor_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            max_features=0.7,
            n_jobs=-1,
            random_state=42,
            verbose=2,
            warm_start=False
        )

        # --- Train regressor ---
        regressor_model.fit(X_train_norm, Y_train_norm)

        # --- Review Feature Importance ---
        # Create DataFrame with feature importances
        filename = 'trial_1_downsampled_aligned_data_(' + phase + ')_(' + time_steps + ').csv'
        df = pd.read_csv(os.path.join(BASE_PATH, filename))
        df = df.iloc[:, :-n_output_cols]        # simply drop action columns
        df = df.iloc[:, 1:]         # drop timevector column

        feat_df = pd.DataFrame({
            "feature": df.columns,                # column names from your data
            "importance": regressor_model.feature_importances_
        })

        # Sort by importance descending
        feat_df = feat_df.sort_values(by="importance", ascending=False)
        top = feat_df.head(20)
        print("\n\nTop Features:\n", top, '\n\n')


    elif regressor == 'mlp':

        # =============================== MULTI LINEAR PERCEPTRON =========================
        # --- Initialize MLP ---
        regressor_model = MLPRegressor(
            hidden_layer_sizes=(50,50,50),  # two hidden layers with 50 neurons each
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            # learning_rate_init=0.00001,
            max_iter=2000,
            early_stopping=True,            # it automatically takes 10% of data for validation
            n_iter_no_change=50,            
            verbose=True
        )

        # --- Train MLP ---
        regressor_model.fit(X_train_norm, Y_train_norm)

        # --- Plot loss curve ---
        plt.figure(figsize=(8,5))
        plt.plot(regressor_model.loss_curve_, label="Training Loss")

        if hasattr(regressor_model, "validation_scores_"):
            # validation_scores_ are R^2 scores per epoch
            val_loss = [1 - v for v in regressor_model.validation_scores_]  # convert R^2 -> pseudo-loss
            plt.plot(val_loss, label="Validation (1 - R2)")

        plt.xlabel("Epoch")
        plt.ylabel("Loss / 1-R2")
        plt.title(f"MLP Training and Validation Progress \n{suffix}")
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(DESTINATION_PATH, 'loss_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()          

    # Save model's results:       
    df_train=pd.DataFrame(train_trials, columns=['trial_id'])
    df_train.to_csv(os.path.join(DESTINATION_PATH,'train_trials.csv'), index=False)
    df_test=pd.DataFrame(test_trials, columns=['trial_id'])
    df_test.to_csv(os.path.join(DESTINATION_PATH, 'test_trials.csv'), index=False)

    model_name = regressor + suffix + '.joblib'
    Xmean_name = regressor + '_Xmean' + suffix + '.npy'
    Xstd_name = regressor + '_Xstd' + suffix + '.npy'
    Ymean_name = regressor + '_Ymean' + suffix + '.npy'
    Ystd_name = regressor + '_Ystd' + suffix + '.npy'

    
    with open(os.path.join(DESTINATION_PATH, model_name), "wb") as f:
        # pickle.dump(regressor_model, f)   
        joblib.dump(regressor_model, f)

    np.save(os.path.join(DESTINATION_PATH, Xmean_name), x_mean)
    np.save(os.path.join(DESTINATION_PATH, Xstd_name), x_std)
    np.save(os.path.join(DESTINATION_PATH, Ymean_name), y_mean)
    np.save(os.path.join(DESTINATION_PATH, Ystd_name), y_std)

    # --- 5. Predict on validation set ---
    Y_pred = regressor_model.predict(X_test_norm)
    # Denormalize predictions
    Y_pred = Y_pred * y_std + y_mean

    # Evaluate model on validation set
    # Evaluate (per output column)
    for i in range(Y_test.shape[1]):
        mse = mean_squared_error(Y_test[:, i], Y_pred[:, i])
        r2 = r2_score(Y_test[:, i], Y_pred[:, i])
        print(f"Output column {i}: MSE={mse:.4f}, R2={r2:.4f}")


def main():

    learn(regressor='mlp', phase='phase_3_pick', time_steps='2_timesteps')


if __name__ == '__main__':
    main()
