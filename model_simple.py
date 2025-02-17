import torch 
import torch.nn as nn
import torch.nn.functional as F

class Conv1D_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvNet(nn.Module): 
    def __init__(self, num_classes=10, dropout=0.0):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
           nn.ZeroPad2d((15,15,0,0)),
           nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = (1,5), stride = (1,1), padding = 0),
           nn.LeakyReLU(),
           nn.Dropout(p=dropout))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 20, out_channels = 40, kernel_size = (2,1), stride = (2,1), padding = 0),
            nn.BatchNorm2d(40, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = (1,3), stride = (1,2))
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels = 80, kernel_size = (1,21), stride = (1,1)),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)))
        self.layer4 = nn.Sequential(
            #nn.ZeroPad2d((15,15,0,0)),
            nn.Conv2d(in_channels=80, out_channels = 160, kernel_size = (1,11), stride = (1,1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)))
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 160, out_channels = 160, kernel_size = (2,1), stride=(7,1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU())
        self.pool4 = nn.Sequential(nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)))
        self.linear1 = nn.Sequential(nn.Linear(800, num_classes),nn.LogSoftmax())
        
            
    def forward(self, x):
        x=x.reshape(-1,1,5,256)
        out = self.layer1(x)
        out = self.layer2(out)
        out= self.layer3(out)
        out = self.pool2(out)
        out = self.layer4(out)
        out = self.pool3(out)
        out = self.layer5(out)
        out = self.pool4(out)
        out = torch.flatten(out,start_dim=1)
        out= self.linear1(out)
        return out




class Conv_multiple_path_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Time pathway
        self.conv1_time = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, padding=1)
        self.pool1_time = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2_time= nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool2_time = nn.MaxPool1d(kernel_size=2, stride=2)

        # Channel pathway
        
        self.conv1_channel=nn.Conv1d(in_channels=256, out_channels=32, kernel_size=1, padding=0)
        # self.pool1_channel = nn.MaxPool1d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1184, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self,x):
        # Time pathway
        out1 = self.relu(self.conv1_time(x))
        out1 = self.pool1_time(out1)
        out1=self.relu(self.conv2_time(out1))
        out1=self.pool2_time(out1)
        out1=out1.view(x.size(0),-1)
        # Channel pathway
        x_T=torch.transpose(x,1,2)
        out2 = self.relu(self.conv1_channel(x_T))
        out2=out2.view(x.size(0),-1)
        # out2 = self.pool1_channel(out2)

        out=torch.cat((out1,out2),1)
        out=self.relu(self.fc1(out))
        out=self.fc2(out)
        return out

class CNN_LSTM(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.lstm = nn.LSTM(input_size=input_shape[0] // 4, hidden_size=80, batch_first=True)  # input_shape[0] should be divided by total pooling kernel_size 
        self.fc1 = nn.Linear(80, 80)
        self.fc2 = nn.Linear(80, num_classes)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # PyTorch expects input as (batch_size, channels, seq_length)
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        
        x = x.transpose(1, 2)  # LSTM expects input as (batch_size, seq_length, input_size)
        import ipdb; ipdb.set_trace()
        x, _ = self.lstm(x)

        x = x[:, -1, :]  # Taking the last output of the LSTM
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x