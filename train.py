import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from util import *
from model import *
import wandb

# Use parser here for the arguments

BATCH_SIZE=36
LR=0.0001
EPOCHS=50
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# run = wandb.init(
#     # Set the project where this run will be logged
#     project="EEG-classifier",
#     # Track hyperparameters and run metadata
#     config={
#         "learning_rate": LR,
#         "Batch size": BATCH_SIZE,
#         "epochs": EPOCHS,
#         "dataset": "EEG_MNIST_INSIGHT"
#     },
# )



csv_file = 'test_IN.csv'  # Replace with the path to your CSV file
test_dataset = MindBigDataDataset(csv_file)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

csv_file='train_IN.csv'
train_dataset=MindBigDataDataset(csv_file)
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=False)

# model=Conv1D_classifier()
# model=ConvNet()
# model=Conv_multiple_path_classifier()
model=CNN_LSTM(input_shape=(10,12),num_classes=10)
model.to(DEVICE)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=LR)


for epoch in range(EPOCHS):
    loss_train=0
    accuracy_train=0
    model.train()
    for batch in train_loader:
        features=batch[0].float().to(DEVICE)
        # import ipdb; ipdb.set_trace()
        labels=batch[1].to(DEVICE)
        # Forward pass
        outputs = model(features)
        predicted_class=torch.argmax(outputs, dim=1)
        accuracy_train+=torch.sum(predicted_class==labels)
        loss = criterion(outputs, labels)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train+=loss.detach().item()
    accuracy_train=accuracy_train/len(train_dataset)

    print(f'Epoch [{epoch+1}/{EPOCHS}], Training Loss: {loss_train/len(train_loader):.4f}, Train accuracy:{accuracy_train}')

    model.eval()
    loss_val=0
    accuracy_test=0
    for batch in test_loader:
        with torch.no_grad():
            features=batch[0].float().to(DEVICE)
            labels=batch[1].to(DEVICE)
            outputs=model(features)
            loss=criterion(outputs,labels)
            predicted_class=torch.argmax(outputs, dim=1)
            accuracy_test+=torch.sum(predicted_class==labels)
            loss_val+=loss.detach().item()
    accuracy_test=accuracy_test/len(test_dataset)
            
    print(f'Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {loss_val/len(test_loader):.4f}, Val accuracy: {accuracy_test}')
    # wandb.log({"Train accuracy": accuracy_train, "Train loss": loss_train/len(train_loader), "Validation accuracy": accuracy_test, "Validation loss": loss_val/len(test_loader)})











