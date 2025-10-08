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
class CSVDataset(Dataset):
    def __init__(self, csv_file, t_cols:List[str], x_cols:List[str], standardize = True):
        super(CSVDataset, self).__init__()
        self.standardize = standardize

        self.path = os.path.join(DIR, csv_file)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"file can not be found: {self.path}")
        
        self.data = pd.read_csv(self.path)
        self.t = self.data[t_cols].values
        self.x = self.data[x_cols].values

        for col in t_cols:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in CSV file")
        for col in x_cols:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in CSV file")

        self.scaler = StandardScaler()
        if self.standardize:
            self.scaler.fit(self.x)
            self.x = self.scaler.transform(self.x)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        t = torch.tensor(self.t[idx], dtype=torch.float32)
        x = torch.tensor(self.x[idx], dtype=torch.float32)

        return t, x
    
    def get_scaler(self):
        if self.standardize:
            return self.scaler
        else:
            raise ValueError("no standardize, no scaler")
    
    def get_data(self):
        return self.data

# %%
# %%
def get_dataloader_from_csv_file(csv_file, t_cols: List[str], x_cols: List[str], 
                                 batch_size, train_ratio=0.9, seed=42):
    """
    Load CSV data and split into train and eval DataLoaders.
    
    Returns:
        train_loader, eval_loader, scaler
    """
    # Load full dataset
    full_dataset = CSVDataset(csv_file, t_cols, x_cols, standardize=True)
    scaler = full_dataset.get_scaler()
    
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    eval_size = total_size - train_size

    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Randomly split indices
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]

    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    eval_dataset = torch.utils.data.Subset(full_dataset, eval_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader, scaler

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