import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import FastICA


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.33 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

fs = 128.0 # sampling frequency
lowcut = 0.1
highcut = 30.0

csv_file='train_IN.csv'
data = pd.read_csv(csv_file)
features = data.iloc[:, 1:].values.astype(np.float32)



filtered_data=None
normalized_data=None
temp=None
S_=None
for feature in features:
    
    temp=feature=feature.reshape(5,256)
    filtered_data = np.array([bandpass_filter(channel, lowcut, highcut, fs) for channel in feature])
    ica = FastICA(n_components=5)
    S_ = ica.fit_transform(filtered_data.T) 
    normalized_data = (filtered_data - np.mean(filtered_data, axis=1, keepdims=True)) / np.std(filtered_data, axis=1, keepdims=True)
    break

import ipdb;ipdb.set_trace()

# Plot the raw and filtered data for the first channel
for channel in range(5):
    plt.figure(figsize=(15, 5))
    plt.subplot(3, 1, 1)
    plt.plot(temp[channel, :])
    plt.title(f'Raw Data - Channel {channel}')
    plt.subplot(3, 1, 2)
    plt.plot(filtered_data[channel, :])
    plt.title(f'Filtered Data - Channel {channel}')
    plt.subplot(3, 1, 3)
    plt.plot(normalized_data[channel, :])
    plt.title(f'Filtered Data - Channel {channel}')
    plt.show()