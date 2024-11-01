
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

batch_size = 20
num_classes = 9
learning_rate = 0.001
num_epochs = 100


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_transforms = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])

imagenet_data = torchvision.datasets.ImageFolder('./data/',transform=all_transforms)
train_size=int(0.85*len(imagenet_data))
test_size=len(imagenet_data)-train_size

train_dataset, val_dataset = random_split(imagenet_data, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)
test_loader=torch.utils.data.DataLoader(dataset =val_dataset ,
                                           batch_size = batch_size,
                                           shuffle = True)

model = ConvNeuralNet()
model=model.to(device)

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.01)  

total_step = len(train_loader)

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):  
        
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

torch.save(model.state_dict(),'./model.pth')

test_acc=0
model.eval() 
  
with torch.no_grad(): 
    
    for i, (images, labels) in enumerate(test_loader): 
          
        images = images.to(device) 
        y_true = labels.to(device) 
          
         
        outputs = model(images) 
          
         
        _, y_pred = torch.max(outputs.data, 1) 
          
         
        test_acc += (y_pred == y_true).sum().item() 
      
    print(f"Test set accuracy = {100 * test_acc / len(val_dataset)} %")

