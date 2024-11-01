import cv2
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
batch_size = 20
num_classes = 9

class ConvNeuralNet(nn.Module):

    def __init__(self): 
        super().__init__() 
        self.model = torch.nn.Sequential( 
            
            torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1),  
            torch.nn.ReLU(), 
            
            torch.nn.MaxPool2d(kernel_size=2), 
  
             
            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1), 
            torch.nn.ReLU(), 
             
            torch.nn.MaxPool2d(kernel_size=2), 
              
            
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1), 
            torch.nn.ReLU(), 
             
            torch.nn.MaxPool2d(kernel_size=2), 
  
            torch.nn.Flatten(), 
            torch.nn.Linear(64*4*4, 512), 
            torch.nn.ReLU(), 
            torch.nn.Linear(512, 10) 
        ) 
  
    def forward(self, x): 
        return self.model(x) 
    

all_transforms = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])

imagenet_data = torchvision.datasets.ImageFolder(root='./data/',transform=all_transforms)
train_loader = torch.utils.data.DataLoader(dataset = imagenet_data,
                                           batch_size = batch_size,
                                           shuffle = True)
classes = imagenet_data.classes
print(classes)
model = ConvNeuralNet()
model.load_state_dict(torch.load("./model.pth"))
model.eval()
z = cv2.imread("26.jpg")
z = cv2.resize(z, (32, 32))
z = TF.to_tensor(z)
z=z.unsqueeze_(0)
output=model(z)
print(output)
#pred=torch.argmax(output,dim=1)
_, predicted_class = torch.max(output,1)
print(predicted_class)
#print(pred)
print('Predicted: ', ' '.join(f'{classes[predicted_class.item()]:5s}'))
