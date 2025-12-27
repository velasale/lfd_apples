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

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader



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
    linear_max = 2      # m/s
    angular_max = 2      # rad/s
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


class VelocityMLP(nn.Module):
    def __init__(self, input_dim, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class VelocityLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])



def create_sequences(X, Y, seq_len):
    """
    X: (N, input_dim)
    Y: (N, output_dim)
    Returns:
        X_seq: (N - seq_len + 1, seq_len, input_dim)
        Y_seq: (N - seq_len + 1, output_dim)
    """
    X_seq, Y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i + seq_len])
        Y_seq.append(Y[i + seq_len - 1])  # predict last step
    return np.array(X_seq), np.array(Y_seq)




def reshape_for_lstm(X, Y, n_timesteps):
    n_samples, n_features = X.shape
    n_sequences = n_samples // n_timesteps
    if n_sequences == 0:
        raise ValueError("Not enough samples for the given number of timesteps")

    X_trimmed = X[:n_sequences * n_timesteps, :]
    Y_trimmed = Y[:n_sequences * n_timesteps, :]

    # Correct: each sequence has n_timesteps rows of all features
    X_seq = X_trimmed.reshape(n_sequences, n_timesteps, n_features)

    # Target: last timestep
    Y_seq = Y_trimmed.reshape(n_sequences, n_timesteps, -1)[:, -1, :]

    return X_seq, Y_seq





def smooth_velocity_loss(v_pred, v_gt, lamda_smooth=0.1):

    # Tracking loss
    loss_track = torch.mean((v_pred-v_gt)**2)

    # Temporal smoothness loss
    dv_pred = v_pred[:, 1:] - v_pred[:, :-1]
    loss_smooth = torch.mean(dv_pred**2)

    return loss_track + lamda_smooth * loss_smooth



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
    BASE_DIRECTORY = os.path.join(BASE_SOURCE_PATH, '05_IL_preprocessed_(memory)')
    experiment = 'experiment_1_(pull)'    
   
    suffix = '_' + experiment + '_' + phase + '_' + time_steps       
    BASE_PATH = os.path.join(BASE_DIRECTORY, experiment, phase, time_steps)

    DESTINATION_DIRECTORY = os.path.join(BASE_SOURCE_PATH, '06_IL_learning')
    DESTINATION_PATH = os.path.join(DESTINATION_DIRECTORY, experiment, phase, time_steps)
    os.makedirs(DESTINATION_PATH, exist_ok=True)

    # Split data into Training trials and Test trials
    all_data, all_trials_paths = load_data(BASE_PATH)    
    cols = all_data.shape[1]

    # Load actions
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
        all_trials_paths, test_size=0.15, shuffle=True)
    
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
            # min_samples_leaf=5,
            # max_features=0.7,
            n_jobs=-1,
            random_state=42,
            verbose=2,
            warm_start=False
        )

        # --- Train regressor ---
        regressor_model.fit(X_train_norm, Y_train_norm)

        # --- Review Feature Importance ---
        # Create DataFrame with feature importances
        filename = 'trial_1_downsampled_aligned_data_transformed_(' + phase + ')_(' + time_steps + ').csv'
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

        # Save feature importances to CSV
        feat_df.to_csv(os.path.join(DESTINATION_PATH, 'rf_feature_importances.csv'), index=False)


    elif regressor == 'mlp':

        # =============================== MULTI LINEAR PERCEPTRON =========================
        # --- Initialize MLP ---
        regressor_model = MLPRegressor(
            hidden_layer_sizes=(50,50),  # two hidden layers with 50 neurons each
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            # learning_rate_init=0.001,
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


    elif regressor == 'mlp_torch':

        print("\n=== Training PyTorch MLP with smoothness loss ===\n")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------- Split training into train/validation ----------------      
        X_train_final, X_val_arr, Y_train_final, Y_val_arr = train_test_split(
            X_train_norm, Y_train_norm, test_size=0.15, shuffle=True, random_state=42
        )

        # Convert to torch tensors
        X_train_t = torch.tensor(X_train_final, dtype=torch.float32).to(device)
        Y_train_t = torch.tensor(Y_train_final, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val_arr, dtype=torch.float32).to(device)
        Y_val_t = torch.tensor(Y_val_arr, dtype=torch.float32).to(device)

        # ---------------- Initialize model ----------------
        regressor_model = VelocityMLP(
            input_dim=X_train_norm.shape[1],
            output_dim=Y_train_norm.shape[1]
        ).to(device)

        optimizer = torch.optim.Adam(regressor_model.parameters(), lr=1e-3)
        n_epochs = 500
        batch_size = 128
        lambda_smooth = 0.2

        train_losses = []
        val_losses = []

        # ---------------- Training loop ----------------
        regressor_model.train()
        N = X_train_t.shape[0]

        for epoch in range(n_epochs):
            epoch_loss = 0.0

            # Do NOT shuffle to keep temporal adjacency
            for i in range(0, N - batch_size, batch_size):
                x_batch = X_train_t[i:i + batch_size]
                y_batch = Y_train_t[i:i + batch_size]

                optimizer.zero_grad()
                y_pred = regressor_model(x_batch)

                # Smoothness loss
                loss_track = torch.mean((y_pred - y_batch) ** 2)
                dv_pred = y_pred[1:] - y_pred[:-1]
                loss_smooth = torch.mean(dv_pred ** 2)

                loss = loss_track + lambda_smooth * loss_smooth
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= (N / batch_size)
            train_losses.append(epoch_loss)

            # ---------------- Compute validation loss ----------------
            regressor_model.eval()
            with torch.no_grad():
                y_val_pred = regressor_model(X_val_t)
                val_track = torch.mean((y_val_pred - Y_val_t) ** 2)
                dv_val = y_val_pred[1:] - y_val_pred[:-1]
                val_smooth = torch.mean(dv_val ** 2)
                val_loss = val_track + lambda_smooth * val_smooth
                val_losses.append(val_loss.item())
            regressor_model.train()

            if epoch % 20 == 0:
                print(f"[Epoch {epoch:03d}] Train Loss = {epoch_loss:.6f}, Val Loss = {val_loss.item():.6f}")

        # ---------------- Plot training and validation loss ----------------
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"PyTorch MLP Training vs Validation Loss\n{suffix}")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(DESTINATION_PATH, "mlp_torch_loss.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # ---------------- Save Torch model ----------------
        torch.save(
            regressor_model.state_dict(),
            os.path.join(DESTINATION_PATH, "mlp_torch_model.pt")
        )


    elif regressor == 'lstm':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------- Hyperparameters ----------------
        sequence_length = 3        # TUNE THIS
        batch_size = 128
        learning_rate = 1e-3
        num_epochs = 100

         # ---------------- Train / Val split ----------------
        X_train_final, X_val_arr, Y_train_final, Y_val_arr = train_test_split(
            X_train_norm,
            Y_train_norm,
            test_size=0.15,
            shuffle=False  # IMPORTANT: keep temporal order
        )

        # ---------------- Create sequences ----------------
        X_train_seq, Y_train_seq = create_sequences(
            X_train_final, Y_train_final, sequence_length
        )
        X_val_seq, Y_val_seq = create_sequences(
            X_val_arr, Y_val_arr, sequence_length
        )
        X_test_seq, Y_test_seq = create_sequences(
            X_test_norm, Y_test, sequence_length
        )

        print("Train sequences:", X_train_seq.shape)
        print("Val sequences:", X_val_seq.shape)
        print("Test sequences:", X_test_seq.shape)

        # ---------------- Torch tensors ----------------
        X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
        Y_train_t = torch.tensor(Y_train_seq, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
        Y_val_t = torch.tensor(Y_val_seq, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)

        # ---------------- DataLoaders ----------------
        train_loader = DataLoader(
            TensorDataset(X_train_t, Y_train_t),
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(X_val_t, Y_val_t),
            batch_size=batch_size,
            shuffle=False
        )


        # ---------------- Initialize LSTM model ----------------
        regressor_model = VelocityLSTM(
            input_dim=X_train_seq.shape[2],
            hidden_dim=64,
            output_dim=Y_train_seq.shape[1]
        ).to(device)


        optimizer = torch.optim.Adam(regressor_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):

            # -------- Train --------
            regressor_model.train()
            train_loss = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                preds = regressor_model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * xb.size(0)

            train_loss /= len(train_loader.dataset)

            # -------- Validation --------
            regressor_model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = regressor_model(xb)
                    val_loss += criterion(preds, yb).item() * xb.size(0)

            val_loss /= len(val_loader.dataset)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if epoch % 10 == 0:
                print(f"[Epoch {epoch:03d}] Train: {train_loss:.6f} | Val: {val_loss:.6f}")


         # ---------------- Plot training and validation loss ----------------
        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"LSTM Training vs Validation Loss\n{suffix}")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(DESTINATION_PATH, "lstm_torch_loss.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # ---------------- Save Torch model ----------------
        torch.save(
            regressor_model.state_dict(),
            os.path.join(DESTINATION_PATH, "lstm_torch_model.pt")
        )

        

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

    
    if regressor in ['rf', 'mlp']:
        with open(os.path.join(DESTINATION_PATH, model_name), "wb") as f:
            joblib.dump(regressor_model, f)

    elif regressor == 'mlp_torch':
        # already saved via torch.save inside the mlp_torch block
        pass


    np.save(os.path.join(DESTINATION_PATH, Xmean_name), x_mean)
    np.save(os.path.join(DESTINATION_PATH, Xstd_name), x_std)
    np.save(os.path.join(DESTINATION_PATH, Ymean_name), y_mean)
    np.save(os.path.join(DESTINATION_PATH, Ystd_name), y_std)

    if regressor in ['mlp_torch', 'lstm']:
        regressor_model.eval()
        # X_test_t = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
        with torch.no_grad():
            Y_pred = regressor_model(X_test_t.to(device)).cpu().numpy()

        # Denormalize
        Y_pred = Y_pred * y_std + y_mean

    else:
        Y_pred = regressor_model.predict(X_test_norm)
        Y_pred = Y_pred * y_std + y_mean


    # Evaluate model on validation set
    # Evaluate (per output column)
    for i in range(Y_test_seq.shape[1]):
        mse = mean_squared_error(Y_test_seq[:, i], Y_pred[:, i])
        r2 = r2_score(Y_test_seq[:, i], Y_pred[:, i])
        print(f"Output column {i}: MSE={mse:.4f}, R2={r2:.4f}")


def main():

    regressors = ['lstm']
    phases = ['phase_1_approach', 'phase_2_contact', 'phase_3_pick']

    phases = ['phase_3_pick']

    for regressor in regressors:
        for phase in phases:
            print(f"================== {phase} ===================")
            for t in range(0, 1):
                time_steps = str(t) + '_timesteps'
                print(f"--- {time_steps} ---")                
                learn(regressor=regressor, phase=phase, time_steps=time_steps)
    


if __name__ == '__main__':
    main()
