# lstm_train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt
import joblib
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Custom imports
from lfd_apples.lfd_learning import DatasetForLearning, save_model



class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, pooling='mean'):
        """
        :param input_dim: number of features per timestep
        :param hidden_dim: hidden size of LSTM
        :param output_dim: number of outputs
        :param num_layers: number of LSTM layers
        :param pooling: 'last' or 'mean' - how to reduce sequence to single output
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True
        )
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        assert pooling in ['last', 'mean'], "pooling must be 'last' or 'mean'"
        self.pooling = pooling

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, input_dim)
        :return: (batch_size, output_dim)
        """
        out, (h_n, c_n) = self.lstm(x)  # out: (batch, seq_len, hidden_dim)

        if self.pooling == 'last':
            # Use last timestep output
            out = out[:, -1, :]
        elif self.pooling == 'mean':
            # Average over all timesteps
            out = out.mean(dim=1)
        out = self.fc_1(out)
        out = self.relu(out)
        return self.fc(out)
    


def train(model, train_loader, val_loader, Y_train_mean, Y_train_std, epochs=500, lr=1e-4):
    '''
    Docstring for train
    
    :param model: Description
    :param train_loader: Description
    :param val_loader: Description
    :param Y_train_mean: Description
    :param Y_train_std: Description
    :param epochs: Description
    :param lr: Description
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")


    Y_train_mean = Y_train_mean.to(device)
    Y_train_std = Y_train_std.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5
    # )
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    # Adaptive LR scheduler (reduce LR on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=lr
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for ep in range(epochs):
        
        # --- Training ---
        model.train()
        train_loss = 0.0
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, Yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb = Xb.to(device).float()
                Yb = Yb.to(device).float()  # ensure same dtype
                pred = model(Xb)                           
                val_loss += nn.MSELoss()(pred, Yb).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # --- Scheduler step ---
        scheduler.step(val_loss)

        # --- Early stopping check ---
        patience = 100
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # save best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {ep}. Best val loss: {best_val_loss:.4f}")
                model.load_state_dict(best_model_state)
                break

        if ep % 10 == 0:
            print(f"[Epoch {ep:03d}] Train={train_loss:.4f}  Val={val_loss:.4f}  LR={optimizer.param_groups[0]['lr']:.6f}")

    # Return losses for plotting
    return train_losses, val_losses


def evaluate(model, test_loader, Y_train_mean, Y_train_std):
    device = next(model.parameters()).device
    model.eval()

    Y_train_mean = Y_train_mean.to(device)
    Y_train_std = Y_train_std.to(device)

    Y_true, Y_pred = [], []

    with torch.no_grad():
        for Xb, Yb_norm in test_loader:
            Xb = Xb.to(device)
            Yb_norm = Yb_norm.to(device)

            pred_norm = model(Xb)

            # Denormalize both
            pred = pred_norm * Y_train_std + Y_train_mean
            Yb = Yb_norm * Y_train_std + Y_train_mean

            Y_pred.append(pred.cpu().numpy())
            Y_true.append(Yb.cpu().numpy())

    Y_true = np.vstack(Y_true)
    Y_pred = np.vstack(Y_pred)

    mse = mean_squared_error(Y_true, Y_pred, multioutput="raw_values")
    print("Test MSE per output:", mse)


def lfd_lstm(SEQ_LEN=10, BATCH_SIZE = 4, phase='phase_1_approach', hidden_dim = 64, num_layers = 1):
    
    
    # === Load Data ===
    print('\nLoading Data ...')
    BASE_SOURCE_PATH = '/home/alejo/Documents/DATA'
    lfd_dataset = DatasetForLearning(BASE_SOURCE_PATH, phase, time_steps='0_timesteps', SEQ_LENGTH=SEQ_LEN)

    # Keep track of inputs used during training
    input_keys = lfd_dataset.input_keys[:-1]
    input_keys_subfolder_name = "__".join(input_keys)
    model_subfolder = os.path.join(lfd_dataset.DESTINATION_PATH, input_keys_subfolder_name)
    os.makedirs(model_subfolder, exist_ok=True)
           
    # Tensor datasets
    train_ds = TensorDataset(lfd_dataset.X_train_tensor_norm, lfd_dataset.Y_train_tensor_norm)
    val_ds   = TensorDataset(lfd_dataset.X_val_tensor_norm, lfd_dataset.Y_val_tensor_norm)
    test_ds   = TensorDataset(lfd_dataset.X_test_tensor_norm, lfd_dataset.Y_test_tensor_norm)

     
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    for Xb, Yb in val_loader:
        print("Xb shape:", Xb.shape, "dtype:", Xb.dtype)
        print("Yb shape:", Yb.shape, "dtype:", Yb.dtype)
        print("Yb min/max:", Yb.min().item(), Yb.max().item())
        break

    X_train_mean = lfd_dataset.X_train_tensor_mean
    X_train_std = lfd_dataset.X_train_tensor_std

    Y_train_mean = lfd_dataset.Y_train_tensor_mean
    Y_train_std = lfd_dataset.Y_train_tensor_std

    # Model
    model = LSTMRegressor(
        input_dim =lfd_dataset.X_train_tensor_norm.shape[2],   # number of features
        hidden_dim = hidden_dim,
        output_dim = lfd_dataset.Y_train_tensor_norm.shape[1],
        num_layers = num_layers,
        pooling='last'
    )

    train_losses, val_losses = train(
        model, train_loader, val_loader,
        Y_train_mean, Y_train_std,
        epochs=1000
    )

    # Plot loss
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.legend()
    plt.grid(True)

   
    # Save plot to file
    prefix = str(num_layers) + '_layers_' + str(hidden_dim) + '_dim_' + str(SEQ_LEN) + "_seq_lstm_"

    plot_path = os.path.join(model_subfolder, prefix + "loss_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Loss plot saved at: {plot_path}")

    # Save model
    model_path = os.path.join(model_subfolder, prefix + "model.pth")    
    torch.save(model.state_dict(), model_path)

    # Save losses history
    losses_path = os.path.join(model_subfolder, prefix + "losses.npz")
    np.savez(losses_path, train_losses=np.array(train_losses), val_losses=np.array(val_losses))

    # Save statistics             
    Xmean_name = prefix + '_Xmean' + lfd_dataset.suffix + '.npy'
    Xstd_name = prefix + '_Xstd' + lfd_dataset.suffix + '.npy'
    Ymean_name = prefix + '_Ymean' + lfd_dataset.suffix + '.npy'
    Ystd_name = prefix + '_Ystd' + lfd_dataset.suffix + '.npy'

    variable_names = [Xmean_name, Xstd_name, Ymean_name, Ystd_name]
    variable_values = [X_train_mean,
                       X_train_std,
                       Y_train_mean,
                       Y_train_std]
    
    for name, value in zip(variable_names, variable_values):
            np.save(os.path.join(model_subfolder, name), value)
    

    # Evaluate model
    evaluate(model, test_loader, Y_train_mean, Y_train_std)


if __name__ == '__main__':

    
    phases = ['phase_1_approach']
    hidden_dim_list = [2048]
    num_layers_list = [2]

    seq_lens = [30]

    for phase in phases:
        print(f'\n------------------ {phase}-------------------')

        for SEQ_LEN in tqdm(seq_lens):

            print(f'\n--- Sequences: {SEQ_LEN} ---')

            if phase !='phase_1_approach' and SEQ_LEN > 50:
                break               

            for num_layers in num_layers_list:

                print(f'\n--- Number of Layers: {num_layers} ---')

                for hidden_dim in hidden_dim_list:

                    print(f'\n--- Number of Hidden dim: {hidden_dim} ---')

                    lfd_lstm(SEQ_LEN=SEQ_LEN, BATCH_SIZE=64, phase=phase, hidden_dim = hidden_dim, num_layers = num_layers)
    
        
