import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

class MultiPhysicsDataset(Dataset):
    """Custom dataset for multi-physics simulation data."""
    
    def __init__(self, data_file, normalize=True):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to CSV with columns [params..., thermal, stress, em]
            normalize: Whether to normalize data
        """
        self.data = pd.read_csv(data_file)
        self.normalize = normalize
        
        # Separate inputs and outputs
        # Assuming last 3 columns are thermal, stress, EM outputs
        self.inputs = self.data.iloc[:, :-3].values
        self.thermal = self.data.iloc[:, -3].values.reshape(-1, 1)
        self.stress = self.data.iloc[:, -2].values.reshape(-1, 1)
        self.em = self.data.iloc[:, -1].values.reshape(-1, 1)
        
        # Fit scalers
        self.input_scaler = StandardScaler()
        self.thermal_scaler = StandardScaler()
        self.stress_scaler = StandardScaler()
        self.em_scaler = StandardScaler()
        
        if normalize:
            self.inputs = self.input_scaler.fit_transform(self.inputs)
            self.thermal = self.thermal_scaler.fit_transform(self.thermal)
            self.stress = self.stress_scaler.fit_transform(self.stress)
            self.em = self.em_scaler.fit_transform(self.em)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.inputs[idx], dtype=torch.float32),
                torch.tensor(self.thermal[idx], dtype=torch.float32),
                torch.tensor(self.stress[idx], dtype=torch.float32),
                torch.tensor(self.em[idx], dtype=torch.float32))
    
    def save_scalers(self, save_dir):
        """Save fitted scalers for inference."""
        os.makedirs(save_dir, exist_ok=True)
        pickle.dump(self.input_scaler, open(f'{save_dir}/input_scaler.pkl', 'wb'))
        pickle.dump(self.thermal_scaler, open(f'{save_dir}/thermal_scaler.pkl', 'wb'))
        pickle.dump(self.stress_scaler, open(f'{save_dir}/stress_scaler.pkl', 'wb'))
        pickle.dump(self.em_scaler, open(f'{save_dir}/em_scaler.pkl', 'wb'))
    
    @staticmethod
    def load_scalers(load_dir):
        """Load fitted scalers for inference."""
        scalers = {
            'input': pickle.load(open(f'{load_dir}/input_scaler.pkl', 'rb')),
            'thermal': pickle.load(open(f'{load_dir}/thermal_scaler.pkl', 'rb')),
            'stress': pickle.load(open(f'{load_dir}/stress_scaler.pkl', 'rb')),
            'em': pickle.load(open(f'{load_dir}/em_scaler.pkl', 'rb'))
        }
        return scalers


def create_dataloader(data_file, batch_size=32, test_split=0.2, shuffle=True):
    """Create train/test dataloaders from CSV file."""
    
    Args:
        data_file: Path to CSV file
        batch_size: Batch size for training
        test_split: Fraction for test set
        shuffle: Whether to shuffle data
        
    Returns:
        train_loader, test_loader, dataset
    """
    dataset = MultiPhysicsDataset(data_file, normalize=True)
    
    # Split into train/test
    n_train = int(len(dataset) * (1 - test_split))
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, len(dataset)))
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, test_loader, dataset


def download_from_google_drive(file_id, destination):
    """Download file from Google Drive."""
    try:
        import gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)
        print(f'Downloaded to {destination}')
    except ImportError:
        print('Install gdown: pip install gdown')


def upload_to_google_drive(file_path, folder_id):
    """Upload file to Google Drive (Colab only)."""
    try:
        from google.colab import files
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
        
        gfile = drive.CreateFile({'title': os.path.basename(file_path),
                                   'parents': [{'id': folder_id}]})
        gfile.SetContentFile(file_path)
        gfile.Upload()
        print(f'Uploaded {file_path} to Google Drive')
    except ImportError:
        print('This function works only in Google Colab')
