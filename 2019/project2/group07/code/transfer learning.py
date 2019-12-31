import torch
import torch.nn as nn
from torchvision import transforms 
import numpy as np
from torchvision.datasets import ImageFolder
import time

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
#print(train_dataset.class_to_idx)
#print(len(train_dataset))
print(len(train_loader))

val_dataset = ImageFolder(r'D:\zhenghui\P\val',transform=transform)


val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
print(len(val_loader))
print(len(val_dataset))





import torchvision.models as models

model = models.vgg16(pretrained=True)

# 查看迁移模型细节
print("迁移VGG16:\n", model)

# 对迁移模型进行调整
for parma in model.parameters():
    parma.requires_grad = False

model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.5),
                                      torch.nn.Linear(4096, 4096),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.5),
                                      torch.nn.Linear(4096, 2))

# 查看调整后的迁移模型
print("调整后VGG16:\n", model)

model.cuda()
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.00001)

epoch_n = 5
time_open = time.time()

for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch+1, epoch_n))
    print("-"*10)
    print("Training...")
    model.train(True)    
    
    running_loss = 0.0
    running_corrects = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
            

        # 梯度归零
        optimizer.zero_grad()

        # 计算损失
        loss = loss_f(outputs , labels)
        loss.backward()
        optimizer.step()
            
        # 计算损失和
        running_loss += float(loss)
            
        # 统计预测正确的图片数
        running_corrects += torch.sum(predicted==labels.data)
            
        
        if (i+1)%10==0 :
            print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4F}%".format(i, running_loss/(i+1), 
                                                                              100*running_corrects/(batch_size *(i+1))))
                
    epoch_loss = running_loss * batch_size / len(train_dataset)
    epoch_acc = 100 * running_corrects / len(train_dataset)
        
    # 输出最终的结果
    print("Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss, epoch_acc))
        
# 输出模型训练、参数优化用时
time_end = time.time() - time_open
print(time_end)


with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += torch.sum(predicted==labels.data)
    epoch_acc = 100 * correct / len(val_dataset)
    print("Acc:{:.4f}%".format(epoch_acc))





test_dataset = ImageFolder(r'D:\zhenghui\P\Test\t',transform=transform)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

print(test_dataset.class_to_idx)



with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        if i==0:
            results=predicted.cpu().detach().numpy()
        else:
            results=np.concatenate((results,predicted.cpu().detach().numpy()))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

