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
   
   # Prepare Training Set
    train_arrays = []
    for trial in train_trials_list:      

        df = pd.read_csv(trial)
        df.drop(columns=['timestamp_vector'], inplace=True)
       
        arr = df.to_numpy()
        train_arrays.append(arr)

    train_combined = np.vstack(train_arrays)
    X_train = train_combined[:, :n_input_cols]
    Y_train = train_combined[:, n_input_cols:]

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


    # Normalize features
    X_train_norm, X_test_norm, mean, std = zscore_normalize(X_train, X_test)

    return X_train_norm, Y_train, X_test_norm, Y_test, mean, std



def main():

    # Load Data
    BASE_DIRECTORY = '/media/alejo/IL_data/04_IL_preprocessed_memory'
    experiment = 'experiment_1_(pull)'
    phase = 'phase_2_contact'
    time_steps = '2_timesteps'              # Number of timesteps considered as input data

    suffix = '_' + experiment + '_' + phase + '_' + time_steps       
    BASE_PATH = os.path.join(BASE_DIRECTORY, experiment, phase, time_steps)

    DESTINATION_DIRECTORY = '/media/alejo/IL_data/05_IL_learning'
    DESTINATION_PATH = os.path.join(DESTINATION_DIRECTORY, experiment, phase, time_steps)
    os.makedirs(DESTINATION_PATH, exist_ok=True)

    # Split data into Training trials and Test trials
    all_data, all_trials_paths = load_data(BASE_PATH)
    
    cols = all_data.shape[1]
    output_cols = 7                         # These are the action columns
    input_cols = cols - output_cols  

    # Approach 1: Split data row-wise
    # Normalize data
    # X_train_norm, Y_train, X_test_norm, Y_test, mean, std = prepare_data(all_data, input_cols)
  

    # Approach 2: Split data trial-wise
    train_trials, test_trials = train_test_split(
        all_trials_paths, test_size=0.2, shuffle=True, random_state=42
    )
    X_train_norm, Y_train, X_test_norm, Y_test, mean, std = prepare_data_approach2(train_trials, test_trials, input_cols)
    

    # Linear Regression
    regressor = "rf"           # MLP or RF    

    if regressor == 'rf':

        # ==================== RANDOM FOREST REGRESSOR ======================
            
        # --- Initialize regressor ---
        regressor_model = RandomForestRegressor(
            n_estimators=100,
            warm_start=True,
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        # --- Train regressor ---
        regressor_model.fit(X_train_norm, Y_train)

        # --- Review Feature Importance ---
        # Create DataFrame with feature importances
        filename = 'trial_1_downsampled_aligned_data_(phase_2_contact)_(' + time_steps + ').csv'
        df = pd.read_csv(os.path.join(BASE_PATH, filename))
        df = df.iloc[:, :-7]        # simply drop action columns
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
            hidden_layer_sizes=(50, 50),  # two hidden layers with 50 neurons each
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=1500,
            early_stopping=True,            # it automatically takes 10% of data for validation
            n_iter_no_change=50,
            random_state=42,
            verbose=True
        )

        # --- Train MLP ---
        regressor_model.fit(X_train_norm, Y_train)

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

    model_name = regressor + suffix + '.pkl'
    mean_name = 'mean' + suffix + '.npy'
    std_name = 'std' + suffix + '.npy'

    
    with open(os.path.join(DESTINATION_PATH, model_name), "wb") as f:
        pickle.dump(regressor_model, f)   

    np.save(os.path.join(DESTINATION_PATH, mean_name), mean)
    np.save(os.path.join(DESTINATION_PATH, std_name), std)

    # --- 5. Predict on validation set ---
    Y_pred = regressor_model.predict(X_test_norm)

    # Evaluate model on validation set
    # Evaluate (per output column)
    for i in range(Y_test.shape[1]):
        mse = mean_squared_error(Y_test[:, i], Y_pred[:, i])
        r2 = r2_score(Y_test[:, i], Y_pred[:, i])
        print(f"Output column {i}: MSE={mse:.4f}, R2={r2:.4f}")


if __name__ == '__main__':
    main()
