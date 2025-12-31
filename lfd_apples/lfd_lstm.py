# lstm_train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def load_data():


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



    data = np.load("lstm_data.npz")
    X = data["X"]   # (N, D)
    Y = data["Y"]   # (N, 6)
    return X, Y


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


def train(model, train_loader, val_loader, epochs=100, lr=1e-3):
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
    SEQ_LEN = 3
    BATCH_SIZE = 256

    X, Y = load_data()

    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
        X, Y, test_size=0.3, shuffle=False
    )

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_tmp, Y_tmp, test_size=0.5, shuffle=False
    )

    X_train_s, Y_train_s = create_sequences(X_train, Y_train, SEQ_LEN)
    X_val_s,   Y_val_s   = create_sequences(X_val,   Y_val,   SEQ_LEN)
    X_test_s,  Y_test_s  = create_sequences(X_test,  Y_test,  SEQ_LEN)

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(Y_train_s, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(Y_val_s, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_s, dtype=torch.float32),
        torch.tensor(Y_test_s, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = LSTMRegressor(
        input_dim=X.shape[1],
        hidden_dim=64,
        output_dim=Y.shape[1]
    )

    train(model, train_loader, val_loader, epochs=100)
    evaluate(model, test_loader)



if __name__ == '__main__':
    main()
