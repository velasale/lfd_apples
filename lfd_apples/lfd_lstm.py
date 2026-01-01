# lstm_train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

import pickle

import matplotlib.pyplot as plt
import joblib
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Custom imports
from lfd_apples.lfd_learning import DatasetForLearning, save_model


def create_sequences(X, Y, seq_len):
    X_seq, Y_seq = [], []

    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i + seq_len])
        Y_seq.append(Y[i + seq_len - 1])

    return np.array(X_seq), np.array(Y_seq)


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def train(model, train_loader, val_loader, epochs=500, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for ep in range(epochs):
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                val_loss += criterion(model(Xb), Yb).item()

        val_loss /= len(val_loader)

        if ep % 10 == 0:
            print(f"[Epoch {ep:03d}] Train={train_loss:.4f}  Val={val_loss:.4f}")


def evaluate(model, test_loader):
    device = next(model.parameters()).device
    model.eval()

    Y_true, Y_pred = [], []

    with torch.no_grad():
        for Xb, Yb in test_loader:
            Xb = Xb.to(device)
            pred = model(Xb).cpu().numpy()

            Y_pred.append(pred)
            Y_true.append(Yb.numpy())

    Y_true = np.vstack(Y_true)
    Y_pred = np.vstack(Y_pred)

    mse = mean_squared_error(Y_true, Y_pred, multioutput="raw_values")
    print("Test MSE per output:", mse)


def main():
    SEQ_LEN = 6
    BATCH_SIZE = 256

    phase='phase_1_approach'
    time_steps='0_timesteps'

    # === Load Data ===
    print('\nLoading Data ...')
    BASE_SOURCE_PATH = '/home/alejo/Documents/DATA'
    lfd_dataset = DatasetForLearning(BASE_SOURCE_PATH, phase, time_steps)

    # Split Training data into Train and Validation data      
    X_train_final, X_val_arr, Y_train_final, Y_val_arr = train_test_split(
        lfd_dataset.X_train_norm, 
        lfd_dataset.Y_train_norm, 
        test_size=0.15, 
        shuffle=True
    )   

    # Create sequences
    X_train_s, Y_train_s = create_sequences(X_train_final, Y_train_final, SEQ_LEN)
    X_val_s,   Y_val_s   = create_sequences(X_val_arr, Y_val_arr, SEQ_LEN)
    X_test_s,  Y_test_s  = create_sequences(lfd_dataset.X_test_norm, lfd_dataset.Y_test_norm, SEQ_LEN)

    # Tensor datasets
    train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32),
                             torch.tensor(Y_train_s, dtype=torch.float32))
    val_ds   = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32),
                             torch.tensor(Y_val_s, dtype=torch.float32))
    test_ds  = TensorDataset(torch.tensor(X_test_s, dtype=torch.float32),
                             torch.tensor(Y_test_s, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Model
    model = LSTMRegressor(
        input_dim=X_train_s.shape[2],   # number of features
        hidden_dim=50,
        output_dim=Y_train_s.shape[1]
    )

    train(model, train_loader, val_loader, epochs=500)

    model_path = os.path.join(lfd_dataset.DESTINATION_PATH, "lstm_model.pth")    
    torch.save(model.state_dict(), model_path)
    # Save model
    save_model("lstm", model, lfd_dataset)

    evaluate(model, test_loader)




if __name__ == '__main__':
    main()
