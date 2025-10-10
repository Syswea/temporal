# %%
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from typing import List
from sklearn.preprocessing import StandardScaler

DIR = "Data"

# %%
def get_dataloader_from_csv_file(csv_file, t_cols, x_cols, batch_size, train_ratio=0.9):
    # Load data
    df = pd.read_csv(os.path.join(DIR, csv_file))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the data
    t_raw = df[t_cols].values.astype(np.float32)
    x_raw = df[x_cols].values.astype(np.float32)

    # Time-based split (NO shuffle!)
    total_size = len(df)
    train_size = int(train_ratio * total_size)
    
    t_train, x_train = t_raw[:train_size], x_raw[:train_size]
    t_eval, x_eval = t_raw[train_size:], x_raw[train_size:]

    # Standardize ONLY on train set
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_eval_scaled = scaler.transform(x_eval)

    # Create datasets
    train_dataset = ArrayDataset(t_train, x_train_scaled)
    eval_dataset = ArrayDataset(t_eval, x_eval_scaled)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader, scaler

class ArrayDataset(Dataset):
    def __init__(self, t_array, x_array):
        self.t = torch.from_numpy(t_array)
        self.x = torch.from_numpy(x_array)
    
    def __len__(self):
        return len(self.t)
    
    def __getitem__(self, idx):
        return self.t[idx], self.x[idx]

# %%
def extract_t_and_y_from_loader(eval_loader: DataLoader):
    all_t = []
    all_y = []

    with torch.no_grad():
        for t_batch, x_batch in eval_loader:
            all_t.append(t_batch)
            all_y.append(x_batch)

    # batch
    t_tensor = torch.cat(all_t, dim=0)
    y_tensor = torch.cat(all_y, dim=0)

    # NumPy
    t_array = t_tensor.cpu().numpy().flatten()
    y_array = y_tensor.cpu().numpy().flatten()

    return t_array, y_array