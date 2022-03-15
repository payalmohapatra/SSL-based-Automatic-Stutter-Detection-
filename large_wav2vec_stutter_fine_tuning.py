import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm

#from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import random
##################################################################################################
# First things first! Set a seed for reproducibility.
# https://www.cs.mcgill.ca/~ksinha4/practices_for_reproducibility/
def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
##################################################################################################
writer = SummaryWriter()
writer = SummaryWriter("wav2vec_base_model_large")
writer = SummaryWriter(comment="2D conv layer + 4 FC; large lv60k model ; lr = 1e-5; 1000 epochs; 10 mins data")
##################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device available is', device)
# wav2vec2.0
bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
#bundle = torchaudio.pipelines.WAV2VEC2_LARGE
print("Sample Rate of model:", bundle.sample_rate)

model_wav2vec = bundle.get_model().to(device)
## Convert audio to numpy to wav2vec feature encodings
def conv_audio_data (filename) :
    waveform, sample_rate = torchaudio.load(filename)
    waveform = waveform.to(device)
    if sample_rate != bundle.sample_rate:
        print('Mismatched sample rate')
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    emission, _ = model_wav2vec(waveform)
    emission = emission.cpu().detach().numpy()
    return emission

x_f = []
y_f = []
x_s = []
y_s = []
# get all stutter data
path_stutter  = "/home/payal/wav2vec/Dataset/all_stutter/"
files_stutter = os.listdir(path_stutter)

for filename in glob.glob(os.path.join(path_stutter, '*.wav')):
    stutter_np = conv_audio_data(filename)
    x_s.append(stutter_np)
    y_s.append(1)

# get all fluent data
discarded = 0
#FIXME :: How can I avoid discarding the mismatched samples?
path_fluent  = "/home/payal/wav2vec/Dataset/all_fluent/"
files_fluent = os.listdir(path_fluent)
for filename in glob.glob(os.path.join(path_fluent, '*.wav')):
    fluent_np = conv_audio_data(filename)
    # fluent_np --> (1, 149, 1024)
    if ((np.shape(fluent_np)[0] != 1) |(np.shape(fluent_np)[1] != 149) | (np.shape(fluent_np)[2] != 1024)) :
        discarded += 1
    else:
        x_f.append(fluent_np)
        y_f.append(0)

# Shuffle all data within a class so that we have samples from all podcasts.
random.shuffle(x_f)
random.shuffle(x_s)

# 100 samples each for 10 mins training
x_f_train = x_f[0:100]
y_f_train = y_f[0:100]
x_s_train = x_s[0:100]
y_s_train = y_s[0:100]

# # 100 samples each for 100  mins training
# x_f_train = x_f[0:2000]
# y_f_train = y_f[0:2000]
# x_s_train = x_s[0:2000]
# y_s_train = y_s[0:2000]

# 100 samples each for 10 mins testing
x_f_test = x_f[-100:-1]
y_f_test = y_f[-100:-1]
x_s_test = x_s[-100:-1]
y_s_test = y_s[-100:-1]

# FIXME :: Shuffle this later on so that all classesa re not given sequentially for training
x_train = x_f_train + x_s_train
y_train = y_f_train + y_s_train

x_test = x_f_test + x_s_test
y_test = y_f_test + y_s_test

## Hyper parameters
batch_size = 32
num_epochs = 1000
#learning_rate = 0.0001
learning_rate = 1e-5


# split data and translate to dataloader
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

n_samples_train = np.shape(x_train)[0]
n_samples_test = np.shape(x_test)[0]
print('Number of samples to train = ', n_samples_train)
print('Number of samples to test = ', n_samples_test)

class AudioDataset(Dataset) :
    def __init__(self,x,y, n_samples) :
        # data loading
        self.x = x
        self.y = y 
        self.n_samples = n_samples
        
        
    def __getitem__(self,index) :
        return self.x[index], self.y[index]

    def __len__(self) :    
        return self.n_samples      

train_dataset = AudioDataset(x_train,y_train,n_samples_train)
test_dataset = AudioDataset(x_test,y_test,n_samples_test)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1)

dataiter = iter(train_loader)
data = dataiter.next()
features,labels = data

class StutterNet(nn.Module):
    def __init__(self, batch_size):
        super(StutterNet, self).__init__()
        # input shape = (batch_size, 1, 149,1024)
        # in_channels is batch size
        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            #torch.nn.Dropout(p=0.5)
        )
        # input size = (batch_size, 8, 74, 512)
        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=1, stride=2)
            #torch.nn.Dropout(p=0.25)
        )
        # input size = (batch_size, 16, 37, 256)
        self.layer3 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=1, stride=2)
            #torch.nn.Dropout(p=0.25)
        )
        # # input size = (batch_size, 32, 19, 128)
        # self.layer4 = nn.Sequential(
        #     torch.nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=1, stride=2)
        #     #torch.nn.Dropout(p=0.5)
        # )
        # input size = (batch_size, 16, 10, 64)
        self.flatten = torch.nn.Flatten()
        self.fc1 = nn.Linear(32*19*128,1000, bias=True)
        self.fc2 = nn.Linear(1000,50, bias=True)
        #self.fc3 = nn.Linear(10000,4000, bias=True)
        self.fc4 = nn.Linear(50,2, bias=True)

    
    def forward(self, x):
        #print('Before Layer1',np.shape(x))
        out = self.layer1(x)
        #print('After layer 1',np.shape(out))
        out = self.layer2(out)
        #print('After layer 2',np.shape(out))
        out = self.layer3(out)
        #print('After layer 3',np.shape(out))
        #out = self.layer4(out)
        out  = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        #out = self.fc3(out)
        out = self.fc4(out)
        #print('After final ',np.shape(out))

        log_probs = torch.nn.functional.log_softmax(out, dim=1)

        return log_probs

model = StutterNet(batch_size).to(device)
print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#weighted loss
#criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, 3]).to(device)) # Class 0 is 75% of the total dataset 
#criterion = nn.LogSoftmax()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    n_correct = 0
    for i, (features, labels) in enumerate(train_loader):  
        features = features.to(device)
        labels = labels.to(device)
        labels = torch.reshape(labels,(np.shape(labels)[0],))
        labels = labels.to(torch.int64)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute Training Accuracy
        _, predicted_labels = torch.max(outputs.data, 1)
        n_correct = (labels == predicted_labels).sum()
        acc = 100.0 * n_correct / outputs.shape[0]
        # visualisation
        writer.add_scalar("Loss/train", loss, epoch)  
        writer.add_scalar("Accuracy/train", acc, epoch)  
        
        if (i+1) % 2 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}, Acc : {acc}%')


test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1)
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
   n_correct = 0
   # Compute F1 score, precision and recall
   predicted_stutter = 0
   labels_stutter = 0
   correct_stutter = 0
   i = 0
   final_label = []
   final_predicted = []
   for features, labels in test_loader:
       print(labels)
       print(np.shape(features))
       features = features.to(device)
       labels = labels.to(device)
       outputs = model(features)
       print(np.shape(outputs))
       # max returns (value ,index)
       _, predicted = torch.max(outputs.data, 1)
       print(predicted)
       print(np.shape(predicted))
       label = torch.transpose(predicted, -1, 0)
       predicted = torch.reshape(predicted,(outputs.shape[0],1))
       i = i+1
       print(n_correct)
       final_label.append(label)
       final_predicted.append(predicted)
       for i in range (0, outputs.shape[0]) :
               # F1 score for stutter
            if (predicted[i] == 1) :
                predicted_stutter +=1
            if (labels[i] == 1) :
                labels_stutter +=1   
            if ((predicted[i] == 1) & (labels[i] == 1)):
                correct_stutter +=1
            if (predicted[i] == labels[i]) :
                n_correct = n_correct + 1



acc_test = 100*n_correct/n_samples_test
print(f'Accuracy of the network on test dataset is : {acc_test} %')
recall = correct_stutter/ labels_stutter
precision = correct_stutter / predicted_stutter
f1_score = 2 * precision * recall / (precision + recall)    
print(f'Precision of the network on test dataset is : {precision}')
print(f'Recall of the network on test dataset is : {recall}')
print(f'F1 Score of the network on test dataset is : {f1_score}')
writer.close()