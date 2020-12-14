
from abc import ABC
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from sklearn import metrics
from data_processing import *
from model import ResNet

data_path = '/Users/fanglinjiajie/locals/datasets/semiconductor/data/'

N_epoch = 10
Batch_size = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data_x, data_y = torch.split(torch.FloatTensor(TRAIN_X), Batch_size), \
#                  torch.split(torch.FloatTensor(TRAIN_Y), Batch_size)

DefectDetector = ResNet(1, 1)
model = DefectDetector

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epoch = 0

recorder = {'training_loss': [],
            'train_accuracy': [],
            }


def train(model, device, optimizer, data_x, data_y, epoch):
    model.train()
    avg_loss = 0
    correct = 0
    x_split, y_split = torch.split(data_x, Batch_size), torch.split(data_y, Batch_size)
    count = 0
    for batch_idx, (data, target) in enumerate(zip(x_split, y_split)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target.unsqueeze(1))
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

        pred = (output > 0.5).float()
        correct += pred.eq(target.view_as(pred)).sum().item()
        avg_loss += len(data) * loss.item()

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, count, len(data_x), 100. * count / len(data_x), loss.item()))
        count += len(data)
    avg_loss /= len(data_x)
    recorder['training_loss'].append(avg_loss)
    recorder['train_accuracy'].append(correct / len(data_x))
    print('Epoch: {} Train Accuracy: {}/{} ({:.2f}%)\n  Train Loss: {:.6f}'.format(epoch, correct, len(data_y),
                                                                                   100. * correct / len(data_y),

                                                                                   avg_loss))
model.load_state_dict(torch.load('DefectDetector', map_location='cpu'))


"""
Prediction
"""
model.eval()
prediction = pd.read_csv(data_path + 'submission_sample.csv')

for i in tqdm(prediction.index):
    x = imgs2torch([prediction.loc[i, 'id'] + '.bmp'], path=test_path)
    prediction.loc[i, 'defect_score'] = model(x).detach().item()



"""
AUC ROC
"""

x, y = boostrap_training_data(100)

model.eval()
y = y.cpu().numpy()
pred_y = model(x).view(-1).detach().numpy()


fpr, tpr, threshold = metrics.roc_curve(y, pred_y)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


"""
Manually check prediction
"""
file = random.choice(test_list)
img = np.array(Image.open(PATH + 'test/' + file))[np.newaxis, :, :]
defect_area = get_defect_area(file)
imgshow(img, defect_area)

x,y = boostrap_training_data(2)

DefectDetector.load_state_dict(torch.load('DefectDetector', map_location='cpu'))


