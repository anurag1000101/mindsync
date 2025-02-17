import os
# import numpy as np
# import torch
import pickle
# from torch.utils.data import Dataset, DataLoader
# import json
# import matplotlib.pyplot as plt
# from glob import glob
# from transformers import BartTokenizer, BertTokenizer
# from tqdm import tqdm
# from fuzzy_match import match
# from fuzzy_match import algorithims


whole_dataset_dicts = []
dataset_path_task2_v1 = '/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/pickle/task2-NR-1.0-dataset.pickle' 
with open(dataset_path_task2_v1, 'rb') as handle:
    whole_dataset_dicts.append(pickle.load(handle))

print()
for key in whole_dataset_dicts[0]:
    print(f'task2_v1, sentence num in {key}:',len(whole_dataset_dicts[0][key]))
print()
zab=whole_dataset_dicts[0]['ZAB']
print(type(zab),type(zab[0]))

raw=zab[0]['raw_eeg']
print(type(raw))
print(len(raw),len(raw[0]))
print(zab[0]['content'])
# for i in whole_dataset_dicts[0]['raw_eeg']:
#     print(i)