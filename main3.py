# 10/30/2025

import torch

print(f"CUDA available? {torch.cuda.is_available()}")
print(f"CPU available? {torch.cpu.is_available()}")

deviceName=""
# set to false if no gpu, set to true if there is gpu
if(torch.cuda.is_available()):
    deviceName="gpu"
elif(torch.cpu.is_available()):
    deviceName="cpu"
else:
    print("No device available!!!")
    quit()
print(f"Using {deviceName}")


import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms



class HiMom(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatted = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),

        )

    def forward(self, x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits


some_data=[[2,3,4],[3,4,5]]

#model = HiMom().to("cuda")
#model = HiMom().to("cpu")
model = HiMom().to(deviceName)
X = some_data
logits=model(X)
pred_probab=nn.Softmax(dim=1)(logits)
y_pred=pred_probab.argmax(1)
print(f"And my prediction is... {y_pred}")


#calculon huggingface
#classify 2-5 classes
#use accuracy or confusion matrix to test it
# make sure the pipeline is working
# do classification

