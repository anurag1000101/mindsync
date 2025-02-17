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
from models.model_wordlevel import *
from transformers import BartTokenizer
import pickle
from common.utils.data import *
import warnings
from common.metrics import compute_metrics
import copy

''' set random seeds '''
seed_val = 312
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

##################################################################################################
## Important Intializations
##################################################################################################

## intializing the config
config =  ModelConfig_wordlevel()

class BrainPeeker:
    def __init__(self,train_dl:  DataLoader, val_dl: DataLoader, test_dl: DataLoader) -> None:
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.gpu_id  = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        logging.info(f"On GPU {self.gpu_id} having local rank of {self.local_rank}")

        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


        assert self.local_rank != -1, "LOCAL_RANK environment variable not set"
        assert self.gpu_id != -1, "RANK environment variable not set"
        self.n_epochs = config.epochs

        ## intializing all the models now
        ## load from chekp path
        self.load_path = config.preload
        self.model = self.load_model(self.load_path)

        # Load BLEU and ROUGE metrics
        self.bleu_metric = load_metric("bleu")
        self.rouge_metric = load_metric("rouge")

        self.model.module.freeze_bart_parameters(True)
        # self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        # self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.module.parameters()), lr=config.lr1, momentum = 0.9)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.module.parameters()), lr=config.lr1)

        self.bestLoss = 10000000000.0

        ### making directorues to save checkpoints, evaluations etc
        ### making output save folders 
        if self.gpu_id == 0:
            self.tasks = "_".join(config.task_name)
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run_name = f"final-eeg2text_wordLevel_{self.tasks}_{self.timestamp}"
            self.save_model_path = f"eeg2text_wordLevel__{self.tasks}_{self.timestamp}"
            self.root = "/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/models/"
            self.save_model_path = os.path.join(self.root,self.save_model_path)
            self.pth_path = f"{self.save_model_path}/pthFiles"
            self.chkpt_path = f"{self.save_model_path}/best_chkpt"
            self.eval_path = f"{self.save_model_path}/evaluations"
            # Create the folder if it doesn't exist
            if not os.path.exists(self.save_model_path):
                os.makedirs(self.save_model_path)
                os.makedirs(self.pth_path)
                os.makedirs(self.chkpt_path)
                os.makedirs(self.eval_path)
                logging.info(f"models, checkpoints and evaluations will be saved in folder at: '{self.save_model_path}'.")
            ## signing in to wanddb
            load_dotenv()
            secret_value_1 = os.getenv("wandb")

            if secret_value_1 is None:
                logging.error(f"Please set Environment Variables properly for wandb login. Exiting.")
                sys.exit(1)
            else:
                # Initialize Wandb with your API keywandb
                wandb.login(key=secret_value_1)
                self.wandb_run = wandb.init(name = self.wandb_run_name, project="ben10")
                logging.info("Login to wandb succesfull!")
    
    def save_model(self, epoch:int):
        logging.info("Saving the model snapshot.")
        snapshot = {
            "model":self.model.module.state_dict(),
            "epoch":epoch,
        }
        torch.save(snapshot, os.path.join(self.pth_path,f"model_epoch_{epoch%6}"))
        logging.info(f"Snapshot checkpointed successfully at location {self.pth_path} with number {epoch%6}")
    
    def show_require_grad_layers(self):
        logging.info("**/"*100)
        logging.info(' require_grad layers:')
        # sanity check
        for name, param in self.model.module.named_parameters():
            if param.requires_grad:
                logging.info(' ', name)
        logging.info("**/"*100)
    
    def load_model(self, path :str):
        model = MindSync(config).to(self.gpu_id)
        model = DDP(model, device_ids=[self.gpu_id])

        if path is not None:
            snapshot = torch.load(path)
            model.module.load_state_dict(snapshot["model"], strict=False)
            logging.info("Models loaded successfully from the saved path.")

        return model

    def run_epoch_stage1(self, epoch: int):
        if self.gpu_id == 0:
            logging.info(f"Epoch: {epoch}")
        for phase in ['train', 'val']:
            if phase == 'train':
                logging.info(f"GPU {self.gpu_id}, Training now...")
                self.model.train()  # Set model to training mode
            else:
                logging.info(f"GPU {self.gpu_id}, Evaluating now...")
                self.model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            if phase == 'train':
                # Disable tqdm on all nodes except the rank 0 GPU on each server
                batch_iterator = tqdm(self.train_dl, desc=f"Processing Train: Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)
            else:
                batch_iterator = tqdm(self.val_dl, desc=f"Processing Val: Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)

            gts = []
            pred = []
            for _, target_string, target_ids, target_mask, word_level_embed, orig_seq_len, input_attn_mask, input_attn_mask_invert  in batch_iterator:
                # raw_eeg = raw_eeg.to(torch.float32).to(self.gpu_id)
                target_ids = target_ids.long().to(self.gpu_id)
                target_mask = target_mask.to(self.gpu_id)
                word_level_embed = word_level_embed.to(torch.float32).to(self.gpu_id)
                input_attn_mask = input_attn_mask.to(self.gpu_id)
                input_attn_mask_invert = input_attn_mask_invert.to(self.gpu_id)
                """replace padding ids in target_ids with -100"""
                target_ids[target_ids == self.tokenizer.pad_token_id] = -100

                if phase == "val":
                    with torch.no_grad():
                        loss, out = self.model.module.forward(word_level_embed, target_ids,
                                                      input_attn_mask, 
                                                      input_attn_mask_invert,
                                                      freezeBart=True)
                elif phase=="train":
                    # backward + optimize only if in training phase
                    self.optimizer.zero_grad()  # Zero gradients
                    
                    loss, out = self.model.module.forward(word_level_embed, target_ids,
                                                        input_attn_mask, 
                                                        input_attn_mask_invert,
                                                        freezeBart=True)

                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    self.optimizer.step()
                assert out is None

                running_loss += loss.item()
            
            # metrics = self.calculate_scores(pred, gts, phase)
            metrics = {}

            # Calculate mean loss after the epoch
            if phase == 'train':
                mean_loss = running_loss / len(self.train_dl)
            else:
                mean_loss = running_loss / len(self.val_dl)
            metrics[f"{phase}_loss_stage1"]  = mean_loss

            # Print combined training and validation stats
            if self.gpu_id == 0:
                ## send the metrics to wancdb
                try:
                    # Log metrics to WandB for this epoch
                    wandb.log(metrics)

                except Exception as err:
                    logging.error("Not able to log to wandb, ", err)

                logging.info("*******")
                logging.info('(GPU {}) {} Loss {:.5f}'.format(self.gpu_id , phase, mean_loss))


    def run_epoch_stage2(self, epoch: int):
        logging.info(f"On gpu: {self.gpu_id}")

        if self.gpu_id == 0:
            logging.info(f"Epoch: {epoch}")
        for phase in ['train', 'val']:
            if phase == 'train':
                logging.info(f"GPU {self.gpu_id}, Training now...")
                self.model.train()  # Set model to training mode
            else:
                logging.info(f"GPU {self.gpu_id}, Evaluating now...")
                self.model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            if phase == 'train':
                # Disable tqdm on all nodes except the rank 0 GPU on each server
                batch_iterator = tqdm(self.train_dl, desc=f"Processing Train: Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)
            else:
                batch_iterator = tqdm(self.val_dl, desc=f"Processing Val: Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)

            gts = []
            pred = []
            for _, target_string, target_ids, target_mask, word_level_embed, orig_seq_len, input_attn_mask, input_attn_mask_invert  in batch_iterator:
                # raw_eeg = raw_eeg.to(torch.float32).to(self.gpu_id)
                target_ids = target_ids.long().to(self.gpu_id)
                target_mask = target_mask.to(self.gpu_id)
                word_level_embed = word_level_embed.to(torch.float32).to(self.gpu_id)
                input_attn_mask = input_attn_mask.to(self.gpu_id)
                input_attn_mask_invert = input_attn_mask_invert.to(self.gpu_id)
                """replace padding ids in target_ids with -100"""
                target_ids[target_ids == self.tokenizer.pad_token_id] = -100

                if phase == "val":
                    with torch.no_grad():
                        loss, out = self.model.module.forward(word_level_embed, target_ids,
                                                        input_attn_mask, 
                                                        input_attn_mask_invert)
                elif phase=="train":
                    # backward + optimize only if in training phase
                    self.optimizer.zero_grad()  # Zero gradients
                    
                    loss, out = self.model.module.forward(word_level_embed, target_ids,
                                                        input_attn_mask, 
                                                        input_attn_mask_invert)

                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    self.optimizer.step()

                assert out is not None

                # ## predictions for metrics
                # probs = out.logits.softmax(dim=-1)
                # values, predictions = probs.topk(1)
                # predictions = torch.squeeze(predictions, dim=-1)
                # # print(f'predictions:{predictions} predictions shape:{predictions.shape}')
                # predicted_string = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
                # gts.extend(target_string)
                # pred.extend(predicted_string)

                running_loss += loss.item()
            
            # metrics = self.calculate_scores(pred, gts, phase)
            metrics = {}

            # Calculate mean loss after the epoch
            if phase == 'train':
                mean_loss = running_loss / len(self.train_dl)
            else:
                mean_loss = running_loss / len(self.val_dl)
            metrics[f"{phase}_loss_stage2"]  = mean_loss

            if phase == "val" and self.gpu_id == 0:
                # logging.info(f"meanloss: {mean_loss}, {type(mean_loss)}")
                if mean_loss < self.bestLoss:
                    logging.info("**x**"*100)
                    logging.info(f"Saving the bset model so far with least val loss: previousBest: {self.bestLoss} , current :{mean_loss}")
                    self.bestLoss = mean_loss
                    snapshot = {
                        "model":self.model.module.state_dict(),
                        "epoch":epoch,
                    }
                    torch.save(snapshot, os.path.join(self.chkpt_path,f"bestModel"))
                    logging.info(f"Snapshot best model checkpointed successfully at location {self.chkpt_path}")

            # Print combined training and validation stats
            if self.gpu_id == 0:
                ## send the metrics to wancdb
                try:
                    # Log metrics to WandB for this epoch
                    wandb.log(metrics)

                except Exception as err:
                    logging.error("Not able to log to wandb, ", err)

                logging.info("*******")
                logging.info('(GPU {}) {} Loss {:.5f}'.format(self.gpu_id , phase, mean_loss))


    def eval(self, epoch: int, teacher_forcing: bool = False):
        torch.cuda.empty_cache()
        logging.info(f"GPU {self.gpu_id} Evaluating on Test data")
        self.model.eval()  # Set model to evaluate mode

        if teacher_forcing:
            out_file_path = os.path.join(self.eval_path, f"epoch_{epoch}_onTEST_tf.txt")
        else:
            out_file_path = os.path.join(self.eval_path, f"epoch_{epoch}_onTEST.txt")


        with open(out_file_path, 'w') as f:
            # Disable tqdm on all nodes except the rank 0 GPU on each server
            batch_iterator = tqdm(self.test_dl, desc=f"Processing Test Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)
            gts = []
            pred = []
            for _ , target_string, target_ids, target_mask, word_level_embed, orig_seq_len, input_attn_mask, input_attn_mask_invert  in batch_iterator:
                # raw_eeg = raw_eeg.to(torch.float32).to(self.gpu_id)
                target_ids = target_ids.long().to(self.gpu_id)
                target_mask = target_mask.to(self.gpu_id)
                word_level_embed = word_level_embed.to(torch.float32).to(self.gpu_id)
                input_attn_mask = input_attn_mask.to(self.gpu_id)
                input_attn_mask_invert = input_attn_mask_invert.to(self.gpu_id)
                """replace padding ids in target_ids with -100"""
                target_ids[target_ids == self.tokenizer.pad_token_id] = -100

                # testing on noise
                # word_level_embed=torch.randn_like(word_level_embed).to(torch.float32).to(self.gpu_id)

                if not teacher_forcing:
                    with torch.no_grad():
                        predictions = self.model.module.generate(word_level_embed, 
                                                        target_ids,
                                                        input_attn_mask, 
                                                        input_attn_mask_invert,
                                                        max_length=100, do_sample=False, repetition_penalty=5.0,
                                                    )
                else:
                    with torch.no_grad():
                        _, seq2seqLMoutput = self.model.module.forward(word_level_embed, target_ids,
                                                                    input_attn_mask, 
                                                                        input_attn_mask_invert)
                    
                    logits = seq2seqLMoutput.logits  # bs*seq_len*voc_sz
                    probs = logits.softmax(dim=-1)
                    values, predictions = probs.topk(1)
                    predictions = torch.squeeze(predictions, dim=-1)
                    # print(f'predictions:{predictions} predictions shape:{predictions.shape}')
                predicted_string = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

                gts.extend(target_string)
                pred.extend(predicted_string)

                for str_id in range(len(target_string)):
                    f.write(f'start################################################\n')
                    f.write(f'Predicted: {predicted_string[str_id]}\n')
                    f.write(f'True: {target_string[str_id]}\n')
                    f.write(f'end################################################\n\n\n')

            # Calculate mean loss after the epoch
            if teacher_forcing:
                t = "tf"
            else:
                t = "not_tf"

            # Print combined training and validation stats
            if self.gpu_id == 0:
                try:
                    metrics = compute_metrics(pred, gts, f"{t}_test")
                    ## send the metrics to wancdb
                    try:
                        # Log metrics to WandB for this epoch
                        wandb.log(metrics)

                    except Exception as err:
                        logging.error("Not able to log to wandb, ", err)
                except:
                    logging.error(f"Error {Exception} while calculating the metrics for the evaluation.")

                logging.info("*******")
                logging.info(f"Evaluation with tf: {teacher_forcing} completed on GPU {self.gpu_id}")


    def train(self):
        logging.info("Starting the training!")
        logging.info("*"*100)
        logging.info("Stage 1: VQVAE training Starts")
        logging.info("*"*100)

        self.eval(1)
        self.eval(1, True)

        for epoch in range(self.n_epochs):
            if epoch == config.stage1:
                logging.info("*"*100)
                logging.info("Stage 1: VQVAE training Ends")
                logging.info("*"*100)

                self.model.module.unfreeze_bart_parameters()
                # self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.module.parameters()), lr=config.lr2, momentum = 0.9)
                self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.module.parameters()), lr=config.lr2)

                logging.info("*"*100)
                logging.info("Stage 2: All Train Starts!")
                logging.info("*"*100)
            elif epoch < config.stage1:
                self.run_epoch_stage1(epoch)
            else:
                self.run_epoch_stage2(epoch)
                if self.gpu_id == 0:
                    # saving the model
                    self.save_model(epoch)
                    self.eval(epoch)
                    self.eval(epoch, True)
        logging.info("Training complete")
            
    def run(self):
        self.train()


def prepare_dataloader(dataset: Dataset):
    return DataLoader(
        dataset,
        drop_last=False,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=True)
    )

def main():
    global saved_dataset_path, num_indices, batch_size
    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"


    whole_dataset_dicts = []
    if 'task1' in config.task_name:
        logging.info("Selecting Zuco1.0 SR")
        dataset_path_task1 = '/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/pickle/task1-SR-1.0/task1-SR-1.0-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in config.task_name:
        logging.info("Selecting Zuco1.0 NR")
        dataset_path_task2 = '/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/pickle/task2-NR-1.0/task2-NR-1.0-dataset.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in config.task_name:
        logging.info("Selecting Zuco1.0 TSR")
        dataset_path_task3 = '/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/pickle/task3-TSR-1.0/task3-TSR-1.0-dataset.pickle' 
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in config.task_name:
        logging.info("Selecting Zuco2.0 NR")
        dataset_path_taskNRv2 = '/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/pickle/task2-NR-2.0/task2-NR-2.0-dataset.pickle' 
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    dataset_setting = 'unique_sent'
    subject_choice = 'ALL'
    print(f'![Debug]using {subject_choice}')
    eeg_type_choice = 'GD'
    print(f'[INFO]eeg type {eeg_type_choice}') 
    bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    print(f'[INFO]using bands {bands_choice}')
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, max_len=config.time_len)
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, max_len=config.time_len)
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, max_len=config.time_len)

    train_dataloader = prepare_dataloader(train_set)
    val_dataloader = prepare_dataloader(dev_set)
    test_dataloader = prepare_dataloader(test_set)
    print(f"Length of datasets: \n train: {len(train_dataloader)} \n val: {len(val_dataloader)} \n test:{len(test_dataloader)}")

    model = BrainPeeker(train_dataloader, val_dataloader, test_dataloader)
    ## train
    model.run()
    return

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Setup distributed training
    init_process_group(backend='nccl')

    # Train the model
    main()

    # Clean up distributed training
    destroy_process_group()