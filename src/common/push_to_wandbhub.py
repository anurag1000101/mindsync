################## used Library  ############################################################
import sys
from dotenv import load_dotenv
import os 
import wandb
import logging


load_dotenv()
secret_value_0 = os.getenv("hugging_face")
secret_value_1 = os.getenv("wandb")

if secret_value_1 is None:
    print("Set correct wanndb env variable named wandb access token!")
    sys.exit(0)

# Initialize Wandb with your API keywandb
wandb.login(key=secret_value_1)

offline_model_path = "/nlsasfs/home/nltm-st/sujitk/temp/DreamDiffusion/results/eeg_pretrain/03-04-2024-21-41-10/checkpoints/checkpoint.pth"

run =  wandb.init(name = "pretrained_model", project="eeg2txt")
artifact = wandb.Artifact('model', type='model')
artifact.add_file(local_path=offline_model_path)
run.log_artifact(artifact)
run.finish()
print("Finished uploading the artifact. 1")
print("Succesfully pushed")
