import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.decomposition import FastICA
class MindBigDataDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, 0].values
        self.features = self.data.iloc[:, 1:].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.labels[idx]
        features = self.features[idx].reshape(5, 256)
        filtered_data=preprocess(features)
        # Removed FastICA since it was not able to converge on the data
        # ica = FastICA(n_components=5)
        # filtered_data = ica.fit_transform(filtered_data.T).T 
        # return torch.tensor(features), torch.tensor(label)
        return torch.tensor(filtered_data), torch.tensor(label)

def preprocess(eeg_data):
    fs = 128.0 # sampling frequency
    lowcut = 1.0
    highcut = 30.0
    filtered_data = np.array([bandpass_filter(channel, lowcut, highcut, fs) for channel in eeg_data])
    # normalized_data = (filtered_data - np.mean(filtered_data, axis=1, keepdims=True)) / np.std(filtered_data, axis=1, keepdims=True)
    return filtered_data

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

