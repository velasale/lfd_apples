import os

import pickle

import matplotlib.pyplot as plt
import joblib
import yaml

import pandas as pd
import numpy as np
from pathlib import Path

# Scikit-learning imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Torch imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader



class DatasetForLearning():

    def __init__(self, BASE_SOURCE_PATH, phase, time_steps):

        self.phase = phase
        self.time_steps = time_steps
        self.BASE_SOURCE_PATH = BASE_SOURCE_PATH
        self.clip = False                            # Clip output data?

        self.define_data_paths()                # Paths to csvs and to store results
        self.load_actions()                     # Data outputs (actions) from yaml file
        self.load_data(self.BASE_PATH)          # List of data-csv filepaths
        self.split_data()                       # Split data into train and test sets (trial wise)
        self.prepare_data()                     # Prepares train and testing sets (normalized)
        self.safe_check_output()                # See normalized output and check all variables within similar range


    def define_data_paths(self):

        BASE_DIRECTORY = os.path.join(self.BASE_SOURCE_PATH, '05_IL_preprocessed_(memory)')
        experiment = 'experiment_1_(pull)'    
    
        self.suffix = '_' + experiment + '_' + self.phase + '_' + self.time_steps       
        self.BASE_PATH = os.path.join(BASE_DIRECTORY, experiment, self.phase, self.time_steps)

        DESTINATION_DIRECTORY = os.path.join(self.BASE_SOURCE_PATH, '06_IL_learning')

        self.DESTINATION_PATH = os.path.join(DESTINATION_DIRECTORY, experiment, self.phase, self.time_steps)
        os.makedirs(self.DESTINATION_PATH, exist_ok=True)


    def load_actions(self):
        """
        Load actions from yaml file
        
        :param self: Description
        """
        
        data_columns_path = config_path = Path(__file__).parent / "config" / "lfd_data_columns.yaml"
        with open(data_columns_path, "r") as f:
            cfg = yaml.safe_load(f)    
        
        self.output_cols = cfg['action_cols']       
        self.n_output_cols = len(self.output_cols)     # These are the ouputs (actions)

        
    def load_data(self, BASE_PATH):
        
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

        self.n_total_cols = combined.shape[1]
        self.n_input_cols = self.n_total_cols - self.n_output_cols
        self.csvs_filepaths_list = filepaths
        
        return combined, filepaths


    def split_data(self):

        self.train_trials, self.test_trials = train_test_split(self.csvs_filepaths_list,
                                                               test_size=0.15,
                                                               shuffle=True)    
        
        # Save model's results:       
        df_train=pd.DataFrame(self.train_trials, columns=['trial_id'])
        df_train.to_csv(os.path.join(self.DESTINATION_PATH,'train_trials.csv'), index=False)
        df_test=pd.DataFrame(self.test_trials, columns=['trial_id'])
        df_test.to_csv(os.path.join(self.DESTINATION_PATH, 'test_trials.csv'), index=False)


    def prepare_data(self):
        """
        Docstring for prepare_data_approach2
        
        :param train_trials_list: Description
        :param test_trials_list: Description
        :param n_input_cols: Description
        """
        
        # Prepare sets of arrays
        self.X_train, self.Y_train = self.prepare_trial_set(self.train_trials, self.n_input_cols, clip=self.clip)
        self.X_test, self.Y_test = self.prepare_trial_set(self.test_trials, self.n_input_cols, clip=self.clip)  
        
        # Normalize features
        self.X_train_norm, self.X_test_norm, self.X_train_mean, self.X_train_std = zscore_normalize(self.X_train, self.X_test)
        self.Y_train_norm, self.Y_test_norm, self.Y_train_mean, self.Y_train_std = zscore_normalize(self.Y_train, self.Y_test)    

        # Check for zeroes
        nan_rows = np.isnan(self.X_train_norm).any(axis=1)
        print(f"Rows with NaNs in X_train_nrom: {np.where(nan_rows)[0]}")
        nan_rows = np.isnan(self.Y_train).any(axis=1)
        print(f"Rows with NaNs in Y_train: {np.where(nan_rows)[0]}")

        return self.X_train_norm, self.Y_train_norm, self.X_test_norm, self.Y_test, self.X_train_mean, self.X_train_std, self.Y_train_mean, self.Y_train_std


    def prepare_trial_set(self, set_csv_list, n_input_cols, clip=False):

        # Open csvs, convert int to arrays, and stack
        set_arrays = []
        for trial in set_csv_list:

            df = pd.read_csv(trial)
            df = df.apply(pd.to_numeric, errors='coerce')

            df.drop(columns=['timestamp_vector'], inplace=True)

            arr = df.to_numpy(dtype=np.float64)
            set_arrays.append(arr)

        set_combined = np.vstack(set_arrays)

        X = set_combined[:, :n_input_cols]
        Y = set_combined[:, n_input_cols:]

        # Clip actions (outputs) if needed
        # (e.g. Franka Arm joint limits)
        Y_clipped = Y.copy()

        if clip:
            max_linear_velocity = 2     # m/s
            max_angular_velocity = 2    # rad/s

            Y_clipped[:, :3] = np.clip(Y_clipped[:, :3], -max_linear_velocity, max_linear_velocity)
            Y_clipped[:, 3:] = np.clip(Y_clipped[:, 3:], -max_angular_velocity, max_angular_velocity)

        return X, Y_clipped
    

    def safe_check_output(self):
        """
        Docstring for safe_check_output
        
        :param self: Description
        """

        # Save to csv
        Y_train_df = pd.DataFrame(self.Y_train_norm)
        Y_train_df.to_csv(os.path.join(self.DESTINATION_PATH, 'Y_train_normalized.csv'), index=False)

        # Plot distribution of Y_train_norm. All values should be around the same order of magnitude
        plt.figure(figsize=(12, 8))
        for i in range(3):
            # Linear velocities (left column)
            plt.subplot(3, 2, 2*i + 1)
            plt.hist(self.Y_train_norm[:, i], bins=50, alpha=0.7)
            plt.title(f'Linear velocity component {i}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True)

            # Angular velocities (right column)
            plt.subplot(3, 2, 2*i + 2)
            plt.hist(self.Y_train_norm[:, i + 3], bins=50, alpha=0.7)
            plt.title(f'Angular velocity component {i}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.DESTINATION_PATH, 'Y_train_normalized_distribution.png'), dpi=300)
        plt.close()


def zscore_normalize(train_set, test_set, eps=1e-8):
    """
    Z-score normalize features using training set statistics.
    """
    
    # Mean and std ONLY from training set
    train_mean = train_set.mean(axis=0)
    train_std = train_set.std(axis=0) + eps

    # Normalize both sets
    train_set_normalized = (train_set - train_mean) / train_std
    test_set_normalized = (test_set - train_mean) / train_std

    return train_set_normalized, test_set_normalized, train_mean, train_std


class VelocityMLP(nn.Module):
    def __init__(self, input_dim, output_dim=6, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim)
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


def rf_regressor(regressor, dataset_class):
    """
    Random forest bazsed regressor
    
    :param dataset_class: Description
    """

    # Initiliaze class with data
    lfd = dataset_class         

    # Initialize rf regressor
    regressor_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        max_features=0.7,
        n_jobs=-1,        
        verbose=2,
    )       

    # Train rf regressor
    regressor_model.fit(lfd.X_train_norm, lfd.Y_train_norm)

    # Review feature importance    
    filename = 'trial_1_downsampled_aligned_data_transformed_(' + lfd.phase + ')_(' + lfd.time_steps + ').csv'
    df = pd.read_csv(os.path.join(lfd.BASE_PATH, filename))
    df = df.iloc[:, :-lfd.n_output_cols]        # simply drop action columns
    df = df.iloc[:, 1:]                         # drop timevector column

    feat_df = pd.DataFrame({
        "feature": df.columns,                # column names from your data
        "importance": regressor_model.feature_importances_
    })

    # Sort Importance in descending order
    feat_df = feat_df.sort_values(by="importance", ascending=False)
    top = feat_df.head(20)
    print("\n\nTop Features:\n", top, '\n\n')

    # Save feature importances to CSV
    feat_df.to_csv(os.path.join(lfd.DESTINATION_PATH, 'rf_feature_importances.csv'), index=False)

    # Save model
    save_model(regressor, regressor_model, lfd)

    return regressor_model 


def mlp_regressor(regressor, dataset_class):

    # Initiliaze class with data
    lfd = dataset_class        

    # Initialize MLP regressor
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
    regressor_model.fit(lfd.X_train_norm, lfd.Y_train_norm)

    # --- Plot loss curve ---
    plt.figure(figsize=(8,5))
    plt.plot(regressor_model.loss_curve_, label="Training Loss")

    if hasattr(regressor_model, "validation_scores_"):
        # validation_scores_ are R^2 scores per epoch
        val_loss = [1 - v for v in regressor_model.validation_scores_]  # convert R^2 -> pseudo-loss
        plt.plot(val_loss, label="Validation (1 - R2)")

    plt.xlabel("Epoch")
    plt.ylabel("Loss / 1-R2")
    plt.title(f"MLP Training and Validation Progress \n{lfd.suffix}")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(lfd.DESTINATION_PATH, 'loss_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()          

    # Save model
    save_model(regressor, regressor_model, lfd)
    return regressor_model


def torch_mlp_regressor(regressor, dataset_class):
    """
    Torch is convenient compared to scikitlearn when a different loss function is needed
    For instance, if we want to penalize smoothness
    
    :param regressor: Description
    :param dataset_class: Description
    """
    print("\n=== Training PyTorch MLP with smoothness loss ===\n")

    # Initiliaze class with data
    lfd = dataset_class         
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split Training data into Train and Validation data      
    X_train_final, X_val_arr, Y_train_final, Y_val_arr = train_test_split(
        lfd.X_train_norm, 
        lfd.Y_train_norm, 
        test_size=0.15, 
        shuffle=False        
    )

    # Convert to torch tensors
    X_train_t = torch.tensor(X_train_final, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train_final, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_arr, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val_arr, dtype=torch.float32).to(device)

    # ---------------- Initialize model ----------------
    regressor_model = VelocityMLP(
        input_dim = lfd.X_train_norm.shape[1],
        output_dim = lfd.Y_train_norm.shape[1]
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
    plt.title(f"PyTorch MLP Training vs Validation Loss\n{lfd.suffix}")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(lfd.DESTINATION_PATH, "mlp_torch_loss.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ---------------- Save Torch model ----------------
    torch.save(
        regressor_model.state_dict(),
        os.path.join(lfd.DESTINATION_PATH, "mlp_torch_model.pt")
    )   

    # Save model
    save_model(regressor, regressor_model, lfd)    
    return regressor_model


def save_model(model_name, regressor_model, dataset_class): 
   
    lfd = dataset_class

    # Names
    model_filename = model_name + lfd.suffix + '.joblib'
    Xmean_name = model_name + '_Xmean' + lfd.suffix + '.npy'
    Xstd_name = model_name + '_Xstd' + lfd.suffix + '.npy'
    Ymean_name = model_name + '_Ymean' + lfd.suffix + '.npy'
    Ystd_name = model_name + '_Ystd' + lfd.suffix + '.npy'

    # Save model
    if model_name in ['rf', 'mlp']:
        with open(os.path.join(lfd.DESTINATION_PATH, model_filename), "wb") as f:
            joblib.dump(regressor_model, f)

    elif regressor_model == 'mlp_torch':
        # already saved via torch.save inside the mlp_torch block
        pass

    # Save variables
    variable_names = [Xmean_name, Xstd_name, Ymean_name, Ystd_name]
    variable_values = [lfd.X_train_mean, lfd.X_train_std, lfd.Y_train_mean, lfd.Y_train_std]

    for name, value in zip(variable_names, variable_values):
        np.save(os.path.join(lfd.DESTINATION_PATH, name), value)



def learn(regressor='mlp', phase='phase_1_approach', time_steps='2_timesteps'):
    """
    Docstring for learn
    
    :param regressor: 'rf' or 'mlp'  Random Forest or Multi-Layer Perceptron
    :param phase: 'phase_1_approach', 'phase_2_contact', 'phase_3_pick'
    :param time_steps: Description
    """
    
    # === Load Data ===
    print('\nLoading Data ...')
    BASE_SOURCE_PATH = '/home/alejo/Documents/DATA'
    lfd_dataset = DatasetForLearning(BASE_SOURCE_PATH, phase, time_steps)

    # === Train Model ===
    print('\nTraining model ...')
    if regressor == 'rf': regressor_model = rf_regressor(regressor, lfd_dataset)              

    elif regressor == 'mlp': regressor_model = mlp_regressor(regressor, lfd_dataset)
       
    elif regressor == 'mlp_torch': regressor_model = torch_mlp_regressor(regressor, lfd_dataset)

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

    # === Predictions ===       
    print('\nRunning Predictions')
    if regressor in ['mlp_torch', 'lstm']:
        
        Y_test_seq = torch.tensor(lfd_dataset.Y_test, dtype=torch.float32)
        X_test_t = torch.tensor(lfd_dataset.X_test_norm, dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        regressor_model.eval()
        # X_test_t = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
        with torch.no_grad():
            Y_pred = regressor_model(X_test_t.to(device)).cpu().numpy()

        # Denormalize
        Y_pred = Y_pred * lfd_dataset.Y_train_std + lfd_dataset.Y_train_mean

    else:
        Y_test_seq = lfd_dataset.Y_test
        
        Y_pred = regressor_model.predict(lfd_dataset.X_test_norm)
        Y_pred = Y_pred * lfd_dataset.Y_train_std + lfd_dataset.Y_train_mean


    # === Evaluate model on TEST set ===    
    for i in range(Y_test_seq.shape[1]):
        mse = mean_squared_error(Y_test_seq[:, i], Y_pred[:, i])
        r2 = r2_score(Y_test_seq[:, i], Y_pred[:, i])
        print(f"Output column {i}: MSE={mse:.4f}, R2={r2:.4f}")


def main():

    regressors = ['rf', 'mlp','mlp_torch']
    phases = ['phase_1_approach', 'phase_2_contact', 'phase_3_pick']
    regressors = ['mlp_torch']
    phases = ['phase_1_approach']

    for regressor in regressors:
        for phase in phases:
            print(f"================== {phase} ===================")
            for t in range(10,11):
                time_steps = str(t) + '_timesteps'
                print(f"--- {time_steps} ---")                
                learn(regressor=regressor, phase=phase, time_steps=time_steps)    


if __name__ == '__main__':
    main()
