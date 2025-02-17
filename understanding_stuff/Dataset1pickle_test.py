import scipy.io as io
import h5py
import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import math
home_directory = os.path.expanduser("~")

# task = "SR" 
# task_name = 'task1-SR' # 'task1-SR', 'task3-TSR'
# task = "NR"
# task_name = 'task2-NR' # 'task1-SR', 'task3-TSR'
task = "TSR" 
task_name = 'task3-TSR' # 'task1-SR', 'task3-TSR'

rootdir = f"/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/zuco1/task_{str.lower(task)}/Matlab files"

print('##############################')
print(f'start processing ZuCo {task_name}-1.0...')

"""config"""
version = 'v1' # 'old'
# version = 'v2' # 'new'

# task_name = args['task_name']
# directory = args['directory']
# task_name = 
# task_name = 



if version == 'v1':
    # input_mat_files_dir = os.path.join(home_directory,f"datasets/ZuCo/{task_name}/Matlab_files")
    input_mat_files_dir = f"/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/zuco1/task_{str.lower(task)}/Matlab files"
# elif version == 'v2':
#     # new version, mat73 
#     input_mat_files_dir = os.path.join(home_directory,f'datasets/ZuCo/{task_name}/Matlab_files')

task_name = f'{task_name}-1.0'
output_dir = os.path.join(home_directory,f'/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/pickle/{task_name}')
output_name = f'{task_name}-dataset.pickle'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""load files"""
print(input_mat_files_dir)

mat_files = os.listdir(input_mat_files_dir)
mat_files = [os.path.join(input_mat_files_dir,mat_file) for mat_file in mat_files]
mat_files = sorted(mat_files)

if len(mat_files) == 0:
    print(f'No mat files found for {task_name}')
    quit()

dataset_dict = {}
for mat_file in tqdm(mat_files):
    print(mat_file)
    subject_name = os.path.basename(mat_file).split('_')[0].replace('results','').strip()
    dataset_dict[subject_name] = []
    
    if version == 'v1':
        matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']
        # matdata = io.loadmat(mat_file)['sentenceData']
    elif version == 'v2':
        matdata = h5py.File(mat_file,'r')
        # print(matdata)
    # print(matdata)
    for sent in matdata:
        ################################
        print(f'sent.rawData {type(sent.rawData)} {math.isnan(sent.rawData)}: {sent.rawData}')
        # print(f'sent.mean_t1 {type(sent.mean_t1)} {math.isnan(sent.mean_t1)}: {sent.mean_t1}')
        print(f'sent.mean_t1 is None: {sent.mean_t1 is None}')
        print(f'sent.mean_t1 is None: {sent.mean_t1!=sent.mean_t1}')

        if sent.rawData is None:
            quit()
        # print(sent.content)
        # print(sent.rawData)
        ################################
        word_data = sent.word
        if not isinstance(word_data, float):
            # sentence level:
            sent_obj = {'content':sent.content}
            # print(sent_obj)
            sent_obj['sentence_level_EEG'] = {'mean_t1':sent.mean_t1, 'mean_t2':sent.mean_t2, 'mean_a1':sent.mean_a1, 'mean_a2':sent.mean_a2, 'mean_b1':sent.mean_b1, 'mean_b2':sent.mean_b2, 'mean_g1':sent.mean_g1, 'mean_g2':sent.mean_g2}

            sent_obj['raw_eeg'] = sent.rawData  # Extra
            
            # print(f"\n\nsent_obj['raw_eeg']: {sent_obj['raw_eeg']}")

            if task_name == 'task1-SR':
                sent_obj['answer_EEG'] = {'answer_mean_t1':sent.mean_t1, 'answer_mean_t2':sent.answer_mean_t2, 'answer_mean_a1':sent.answer_mean_a1, 'answer_mean_a2':sent.answer_mean_a2, 'answer_mean_b1':sent.answer_mean_b1, 'answer_mean_b2':sent.answer_mean_b2, 'answer_mean_g1':sent.answer_mean_g1, 'answer_mean_g2':sent.answer_mean_g2}
            
            # word level:
            sent_obj['word'] = []
                        
            word_tokens_has_fixation = [] 
            word_tokens_with_mask = []
            word_tokens_all = []

            for word in word_data:
                word_obj = {'content':word.content}
                word_tokens_all.append(word.content)
                # TODO: add more version of word level eeg: GD, SFD, GPT
                word_obj['nFixations'] = word.nFixations
                if word.nFixations > 0:    
                    word_obj['word_level_EEG'] = {'FFD':{'FFD_t1':word.FFD_t1, 'FFD_t2':word.FFD_t2, 'FFD_a1':word.FFD_a1, 'FFD_a2':word.FFD_a2, 'FFD_b1':word.FFD_b1, 'FFD_b2':word.FFD_b2, 'FFD_g1':word.FFD_g1, 'FFD_g2':word.FFD_g2}}
                    word_obj['word_level_EEG']['TRT'] = {'TRT_t1':word.TRT_t1, 'TRT_t2':word.TRT_t2, 'TRT_a1':word.TRT_a1, 'TRT_a2':word.TRT_a2, 'TRT_b1':word.TRT_b1, 'TRT_b2':word.TRT_b2, 'TRT_g1':word.TRT_g1, 'TRT_g2':word.TRT_g2}
                    word_obj['word_level_EEG']['GD'] = {'GD_t1':word.GD_t1, 'GD_t2':word.GD_t2, 'GD_a1':word.GD_a1, 'GD_a2':word.GD_a2, 'GD_b1':word.GD_b1, 'GD_b2':word.GD_b2, 'GD_g1':word.GD_g1, 'GD_g2':word.GD_g2}
                    sent_obj['word'].append(word_obj)
                    word_tokens_has_fixation.append(word.content)
                    word_tokens_with_mask.append(word.content)
                else:
                    word_tokens_with_mask.append('[MASK]')
                    # if a word has no fixation, use sentence level feature
                    # word_obj['word_level_EEG'] = {'FFD':{'FFD_t1':sent.mean_t1, 'FFD_t2':sent.mean_t2, 'FFD_a1':sent.mean_a1, 'FFD_a2':sent.mean_a2, 'FFD_b1':sent.mean_b1, 'FFD_b2':sent.mean_b2, 'FFD_g1':sent.mean_g1, 'FFD_g2':sent.mean_g2}}
                    # word_obj['word_level_EEG']['TRT'] = {'TRT_t1':sent.mean_t1, 'TRT_t2':sent.mean_t2, 'TRT_a1':sent.mean_a1, 'TRT_a2':sent.mean_a2, 'TRT_b1':sent.mean_b1, 'TRT_b2':sent.mean_b2, 'TRT_g1':sent.mean_g1, 'TRT_g2':sent.mean_g2}
                    
                    # NOTE:if a word has no fixation, simply skip it
                    continue
            
            sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
            sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
            sent_obj['word_tokens_all'] = word_tokens_all
            
            # print(sent_obj)
            dataset_dict[subject_name].append(sent_obj)

        else:
            print(f'missing sent: subj:{subject_name} content:{sent.content}, return None')
            dataset_dict[subject_name].append(None)

            continue

        # print(sent_obj)
        # break
    break
    # print(dataset_dict.keys())
    # print(dataset_dict[subject_name][0].keys())
    # print(dataset_dict[subject_name][0]['content'])
    # print(dataset_dict[subject_name][0]['word'][0].keys())
    # print(dataset_dict[subject_name][0]['word'][0]['word_level_EEG']['FFD'])

# """output"""
# if task_name == 'task1-SR':
#     with open(os.path.join(output_dir,'task1-SR-dataset.json'), 'w') as out:
#         json.dump(dataset_dict,out,indent = 4)

# with open(os.path.join(output_dir,output_name), 'wb') as handle:
#     pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print('write to:', os.path.join(output_dir,output_name))


# """sanity check"""
# # check dataset
# with open(os.path.join(output_dir,output_name), 'rb') as handle:
#     whole_dataset = pickle.load(handle)
# print('subjects:', whole_dataset.keys())

# if version == 'v1':
#     print('num of sent:', len(whole_dataset['ZAB']))
#     print()

print('-'*40,end='\n')
