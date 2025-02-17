import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer, BertTokenizer
from tqdm import tqdm
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
# from common.utils.data import *
import warnings



def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std  # Causing NaN values in last channel raw_eeg (std=0)
    return input_tensor 

def normalize_eeg_data(eeg_data):
    """
    Normalize EEG data such that each channel has mean 0 and standard deviation 1.
    
    Args:
    - eeg_data (numpy.ndarray): EEG data array with shape (num_channels, num_samples)
    
    Returns:
    - normalized_data (numpy.ndarray): Normalized EEG data array
    """
    # Calculate mean and standard deviation for each channel
    channel_means = np.mean(eeg_data, axis=1, keepdims=True)
    channel_stds = np.std(eeg_data, axis=1, keepdims=True)
    
    # print(channel_stds)
    # Normalize each channel
    normalized_data = (eeg_data - channel_means) / channel_stds
    
    return normalized_data
global_cnt=0
def get_input_sample(sent_obj, tokenizer, eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], max_len = 56, add_CLS_token = False):
    
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105*len(bands):
            # print(f'expect word eeg embedding dim to be {105*len(bands)}, but got {len(word_eeg_embedding)}, return None')
            return None
        # assert len(word_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)

    def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []
        for band in bands:
            key = 'mean'+band
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105*len(bands) #840
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    if sent_obj is None:  # eg: ZJS
        # print(f'  - skip bad sentence')   
        return None

    input_sample = {}
    # get target label
    target_string = sent_obj['content']
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=120, truncation=True, return_tensors='pt', return_attention_mask = True)

    # print(f"input string: {target_string}, --> tokenized string: {target_tokenized} \n decoded: {tokenizer.decode(target_tokenized)}")
    # input string: Henry Ford, with his son Edsel, founded the Ford Foundation in 1936 as a local philanthropic organization with a broad charter to promote human welfare., --> tokenized string: tensor([    0, 29648,  2493,     6,    19,    39,   979,  2344,  5317,     6,
    #      4790,     5,  2493,  2475,    11, 31025,    25,    10,   400, 14054,
    #       636,  1651,    19,    10,  4007,  9827,     7,  3720,  1050,  6642,
    #         4,     2,     1,     1,     1,     1,     1,     1,     1,     1,
    #         1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
    #         1,     1,     1,     1,     1,     1]) 
    # decoded: <s>Henry Ford, with his son Edsel, founded the Ford Foundation in 1936 as a local philanthropic organization with a broad charter to promote human welfare.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>

    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    input_sample['target_string'] = target_string
    
    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        # print('[NaN sent level eeg]: ', target_string)
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor
    # print(f"Sentence level embedding shape :{sent_level_eeg_tensor.shape}")
    
    print(f"raw_data (not normalized) {np.isnan(sent_obj['raw_eeg']).any()}: {sent_obj['raw_eeg']}")
    # if np.isnan(sent_obj['raw_eeg']).any():
    #     # print('[NaN sent level eeg]: ', target_string)
    #     print(f"raw_data is NaN: {np.isnan(sent_obj['raw_eeg']).any()} {sent_obj['raw_eeg']}")
    #     return None
    raw_data = normalize_eeg_data(sent_obj['raw_eeg'].reshape(105,-1))
    print(f'raw_data (normalized) {np.isnan(raw_data).any()}: {raw_data}')

    raw_data[np.isnan(raw_data)] = 0
    print(f'raw_data (normalized and zeroed) {np.isnan(raw_data).any()}: {raw_data}')

    input_sample['raw_eeg'] = raw_data
####################################################################################
    global_cnt+=1
    if(global_cnt>=5): quit()

####################################################################################
    # get sentiment label
    # handle some wierd case
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty','empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1','film.')
    
    #if target_string in ZUCO_SENTIMENT_LABELS:
    #    input_sample['sentiment_label'] = torch.tensor(ZUCO_SENTIMENT_LABELS[target_string]+1) # 0:Negative, 1:Neutral, 2:Positive
    #else:
    #    input_sample['sentiment_label'] = torch.tensor(-100) # dummy value
    # input_sample['sentiment_label'] = torch.tensor(-100) # dummy value

    # get input embeddings
    word_embeddings = []

    """add CLS token embedding at the front"""
    if add_CLS_token:
        word_embeddings.append(torch.ones(105*len(bands)))

    words = []

    for word in sent_obj['word']:
        # add each word's EEG embedding as Tensors
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands = bands)
        # check none, for v1 dataset
        if word_level_eeg_tensor is None:
            return None
        # check nan:
        if torch.isnan(word_level_eeg_tensor).any():
            # print()
            # print('[NaN ERROR] problem sent:',sent_obj['content'])
            # print('[NaN ERROR] problem word:',word['content'])
            # print('[NaN ERROR] problem word feature:',word_level_eeg_tensor)
            # print()
            return None
            
        # print(f"Shape of each word level embedding: {word_level_eeg_tensor.shape}")
        words.append(word['content'])
        word_embeddings.append(word_level_eeg_tensor)
    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))

    # print(f"After stacking final word level embedding: {torch.stack(word_embeddings).shape}")

    input_sample['input_embeddings'] = torch.stack(word_embeddings) # max_len * (105*num_bands)
    input_sample['target_words_string'] = words


    # mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len) # 0 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask'][:len(sent_obj['word'])+1] = torch.ones(len(sent_obj['word'])+1) # 1 is not masked
    else:
        input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word'])) # 1 is not masked
    

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample['input_attn_mask_invert'] = torch.ones(max_len) # 1 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])+1] = torch.zeros(len(sent_obj['word'])+1) # 0 is not masked
    else:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word'])) # 0 is not masked

    

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])
    
    # clean 0 length data
    if input_sample['seq_len'] == 0:
        # print('discard length zero instance: ', target_string)
        return None

    return input_sample


class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject = 'ALL', eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], setting = 'unique_sent', is_add_CLS_token = False, max_len : int= None):
        self.inputs = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        if not isinstance(input_dataset_dicts,list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')
        
        for input_dataset_dict in input_dataset_dicts:
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print('[INFO]using subjects: ', subjects)
            else:
                subjects = [subject]
            
            total_num_sentence = len(input_dataset_dict[subjects[0]])
            
            train_divider = int(0.8*total_num_sentence)
            dev_divider = train_divider + int(0.1*total_num_sentence)
            
            print(f'train divider = {train_divider}')
            print(f'dev divider = {dev_divider}')

            if setting == 'unique_sent':
                # take first 80% as trainset, 10% as dev and 10% as test
                if phase == 'train':
                    print('[INFO]initializing a train set...')
                    for key in subjects:
                        for i in range(train_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'dev':
                    print('[INFO]initializing a dev set...')
                    for key in subjects:
                        for i in range(train_divider,dev_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'test':
                    print('[INFO]initializing a test set...')
                    for key in subjects:
                        for i in range(dev_divider,total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            elif setting == 'unique_subj':
                print('WARNING!!! only implemented for SR v1 dataset ')
                # subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'] for train
                # subject ['ZMG'] for dev
                # subject ['ZPH'] for test
                if phase == 'train':
                    print(f'[INFO]initializing a train set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH','ZKW']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == 'dev':
                    print(f'[INFO]initializing a dev set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZMG']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == 'test':
                    print(f'[INFO]initializing a test set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZPH']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            # print('++ adding task to dataset, now we have:', len(self.inputs))

        # print('[INFO]input tensor size:', self.inputs[0]['input_embeddings'].size())
        print()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        
        raw_eeg = input_sample['raw_eeg']
        if self.max_len is not None:
            # Truncate or pad raw_eeg array
            if raw_eeg.shape[1] > self.max_len:
                raw_eeg = raw_eeg[:, :self.max_len]
            elif raw_eeg.shape[1] < self.max_len:
                pad_width = ((0, 0), (0, self.max_len - raw_eeg.shape[1]))
                raw_eeg = np.pad(raw_eeg, pad_width, mode='constant', constant_values=0)
        
        # return (
        #     raw_eeg,
        #     input_sample['target_string'],
        #     input_sample['target_ids'], 
        #     input_sample['target_mask'],  
        # )
        return (
            raw_eeg,
            input_sample['target_string'],
            input_sample['target_ids'], 
            input_sample['target_mask'],  
            input_sample['input_embeddings'], # new
            input_sample['seq_len'], # new
            input_sample['input_attn_mask'], 
            input_sample['input_attn_mask_invert'],
        )

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
dataset_setting = 'unique_sent'
subject_choice = 'ALL'
eeg_type_choice = 'GD'
bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 

config=ModelConfig()

def prepare_dataloader(dataset: Dataset):
    return DataLoader(
        dataset,
        drop_last=False,
        batch_size=1,
        shuffle=False,
        # sampler=DistributedSampler(dataset, shuffle=True)
    )


whole_dataset_dicts=[]
print('Creating whole_dataset_dicts...')
dataset_path_task1 = '/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/pickle/task1-SR-1.0/task1-SR-1.0-dataset.pickle' 
with open(dataset_path_task1, 'rb') as handle:
    whole_dataset_dicts.append(pickle.load(handle))

dataset_path_task2 = '/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/pickle/task2-NR-1.0/task2-NR-1.0-dataset.pickle' 
with open(dataset_path_task2, 'rb') as handle:
    whole_dataset_dicts.append(pickle.load(handle))

dataset_path_task3 = '/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/pickle/task3-TSR-1.0/task3-TSR-1.0-dataset.pickle' 
with open(dataset_path_task3, 'rb') as handle:
    whole_dataset_dicts.append(pickle.load(handle))
print('whole_dataset_dicts created')

dset = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, max_len=config.time_len)
dl = prepare_dataloader(dset)
batch_iterator = tqdm(dl)

for raw_eeg, target_string, target_ids, target_mask, word_level_embed, orig_seq_len, input_attn_mask, input_attn_mask_invert  in batch_iterator:
    raw_eeg = raw_eeg.to(torch.float32) #.to(self.gpu_id)
    print(f'raw_eeg.shape: {raw_eeg.shape}, raw_eeg: {raw_eeg}')
    break

# i=1
total_count=nan_raw_eeg_count=0
for raw_eeg, target_string, target_ids, target_mask, word_level_embed, orig_seq_len, input_attn_mask, input_attn_mask_invert  in batch_iterator:
    raw_eeg = raw_eeg.to(torch.float32) #.to(self.gpu_id)
    # print(f'raw_eeg.shape: {raw_eeg.shape}, raw_eeg: {raw_eeg}')
    # print(torch.isnan(raw_eeg).any(), end=', ')
    # if(i%10==0): print()
    # i+=1
    # if(i>=50):
    #     break
    has_nan=torch.isnan(raw_eeg).any().item()  # True/False
    nan_raw_eeg_count+=has_nan
    total_count+=1
print(f'total_count: {total_count}, nan_raw_eeg_count: {nan_raw_eeg_count}')
#########################################################################################################
#########################################################################################################
exit(0)
print('Creating whole_dataset_dicts...')
zuco2 = []
dataset_path_task2_v2 = '/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/pickle/task2-NR-2.0/task2-NR-2.0-dataset.pickle' 
with open(dataset_path_task2_v2, 'rb') as handle:
    zuco2.append(pickle.load(handle))
print('whole_dataset_dicts created')

dset2 = ZuCo_dataset(zuco2, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, max_len=config.time_len)
dl2 = prepare_dataloader(dset2)
batch_iterator2 = tqdm(dl2)

for raw_eeg, target_string, target_ids, target_mask, word_level_embed, orig_seq_len, input_attn_mask, input_attn_mask_invert  in batch_iterator2:
    raw_eeg = raw_eeg.to(torch.float32) #.to(self.gpu_id)
    print(f'raw_eeg.shape: {raw_eeg.shape}, raw_eeg: {raw_eeg}')
    break

total_count=nan_raw_eeg_count=0
for raw_eeg, target_string, target_ids, target_mask, word_level_embed, orig_seq_len, input_attn_mask, input_attn_mask_invert  in batch_iterator2:
    raw_eeg = raw_eeg.to(torch.float32) #.to(self.gpu_id)
    # print(f'raw_eeg.shape: {raw_eeg.shape}, raw_eeg: {raw_eeg}')
    # print(torch.isnan(raw_eeg).any(), end=', ')
    # if(i%10==0): print()
    # i+=1
    # if(i>=50):
    #     break

    has_nan=torch.isnan(raw_eeg).any().item()  # True/False
    nan_raw_eeg_count+=has_nan
    total_count+=1

print(f'total_count: {total_count}, nan_raw_eeg_count: {nan_raw_eeg_count}')
