#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

#### importing necessary modules and libraries
from torch.utils.data import Dataset
from dotenv import load_dotenv
from datasets import load_metric
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
sys.path.append('/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/src')
from dataclasses import dataclass
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
import numpy as np
from torch import optim
import logging
from datetime import datetime
import wandb
from config import ModelConfig
from models.model import *
from transformers import BartTokenizer
import pickle
from common.tsne.data import *
import warnings
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

##################################################################################################
## Important Intializations
##################################################################################################
device = "cuda"

## intializing the config
config = ModelConfig()

metafile  = torch.load("/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/models/eeg2text_-20240407_154417/pthFiles/model_epoch_2", map_location=device)
state_dict = metafile['model']
model = MindSync(config).to(device)


### Prepare dataset
def prepare_dataloader(dataset: Dataset):
    return DataLoader(
        dataset,
        drop_last=False,
        batch_size=config.batch_size,
        shuffle=False,
    )

whole_dataset_dicts = []
dataset_path_task2_v2 = '/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/pickle/task2-NR-2.0/task2-NR-2.0-dataset.pickle' 
with open(dataset_path_task2_v2, 'rb') as handle:
    whole_dataset_dicts.append(pickle.load(handle))
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
dataset_setting = 'unique_sent'
subject_choice = 'ALL'
eeg_type_choice = 'GD'
bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, max_len=config.time_len)

test_dl = prepare_dataloader(test_set)
logging.info(f"Len of test set: {len(test_dl)}")

u, v = model.load_state_dict(state_dict, strict=False)
logging.warning(f"Missing and Unmatched keys: \n{u} \n {v}")

## evaluating and extracting the embeddings
logging.info("Extracting the embeddings!")
model.eval()
batch_iterator = tqdm(test_dl, desc=f"Extracting the quantized embeddings")

embeddings = []
labels = []
for raw_eeg, target_string, target_ids, target_mask, subjects in batch_iterator:
    torch.cuda.empty_cache()

    raw_eeg = raw_eeg.to(torch.float32).to(device)
    target_ids = target_ids.long().to(device)
    target_mask = target_mask.to(device)

    emb = model.getEmbedding(raw_eeg, target_ids)

    # Append embeddings and labels to the respective lists
    embeddings.append(emb.cpu().detach().numpy())
    labels.extend(subjects)

# Concatenate lists to create complete numpy arrays
embeddings_array = np.concatenate(embeddings, axis=0)
# labels_array = np.concatenate(labels, axis=0)
print(f"shape of embeddings: ", embeddings_array.shape)
embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
print(f"After flattenting, {embeddings_array.shape} and labels: \n {labels}")


# Standardize the embeddings
scaler = StandardScaler()
embeddings_array_scaled = scaler.fit_transform(embeddings_array)

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2,  random_state = 42)
embeddings_tsne = tsne.fit_transform(embeddings_array_scaled)

# Save embeddings and labels into a .npz file
np.savez("tsne-subjwise.npz", embeddings=embeddings_tsne, labels=labels)

# Load data
# data = np.load("/Downloads/tsne-hiddenFeatures.npz")
data, y  = embeddings_tsne, labels

data1, y1  = data, y

# Plot t-SNE visualization
# Create pretty plot using Seaborn
plt.figure(figsize=(8, 8))
sns.scatterplot(x=data1[:, 0], y=data1[:, 1], hue=y1, palette='husl', s=200, alpha=0.3)
plt.title('t-SNE Visualization for subject-wise embeddings representation', weight='bold').set_fontsize(14)
plt.xlabel('t-SNE dimension 1', weight='bold').set_fontsize(10)
plt.ylabel('t-SNE dimension 2', weight='bold').set_fontsize(10)
plt.legend(title='Subject', fontsize=10)

# Save the plot
plt.savefig("tsne_plot.png")
plt.show()


