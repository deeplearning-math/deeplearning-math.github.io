import torch
import torch.nn as nn
from torchvision import transforms 
import numpy as np
from torchvision.datasets import ImageFolder

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



batch_size = 50

transform = transforms.Compose(
 [
 transforms.ToTensor(),
 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
 ]
)

train_dataset = ImageFolder(r'D:\zhenghui\P\Train',transform=transform)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
print(train_dataset.class_to_idx)



test_dataset = ImageFolder(r'D:\zhenghui\P\Test\t',transform=transform)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
print(test_dataset.class_to_idx)

import torchvision.models as models

model = models.vgg16(pretrained=True)
model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
#print(model)

model=model.eval()
model.cuda()


for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        outputs = model(images)
#        if i==0:
#            results=outputs
#        else:
#            results=torch.cat((results,outputs),dim=0)

        if i==0:
            results=outputs.cpu().detach().numpy()
            label=labels.numpy()
        else:
            results=np.vstack((results,outputs.cpu().detach().numpy()))
            label=np.concatenate((label,labels.numpy()))


print(len(test_loader))

for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
#        if i==0:
#            results=outputs
#        else:
#            results=torch.cat((results,outputs),dim=0)

        if i==0:
            results=outputs.cpu().detach().numpy()
        else:
            results=np.vstack((results,outputs.cpu().detach().numpy()))


            
            
            
            
            
            