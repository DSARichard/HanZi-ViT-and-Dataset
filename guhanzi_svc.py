import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import SimpleViT
from simple_vit_conv import SimpleViTConv
from overlap_patches import overlap_patches
import os

model_type = ["svc", "vit", "net"][0]
device = torch.device("cuda") if(torch.cuda.is_available()) else torch.device("cpu")

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(1296, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 4)
  
  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

if(model_type == "vit"):
  vit = SimpleViT(
    image_size = 50,
    patch_size = 10,
    num_classes = 4,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    channels = 1,
  )

elif(model_type == "svc"):
  vit = SimpleViTConv(
    image_size = 68,
    patch_size = 17,
    num_classes = 4,
    conv_dim = 8,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    channels = 1,
  )

else:
  vit = LeNet()

vit.to(device)

class CharDataset():
  def __init__(self, data, targets, transform = None):
    self.data = data
    self.targets = targets
    self.transform = transform
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    img = np.expand_dims(self.data[idx], 0)
    label = np.int64(self.targets[idx])
    if(self.transform is not None):
      img = self.transform(img)
    return img, label

train_data = []
for file in sorted(os.listdir("GuHanZi_dataset_bigsplit/trainset")):
  if(not "jpg" in file):
    continue
  file = "GuHanZi_dataset_bigsplit/trainset/" + file
  img = cv2.imread(file, 0)
  img = np.float32(img)/255
  if(model_type == "svc"):
    img = overlap_patches(img, 17, 6)
  train_data.append(img)
train_data = np.array(train_data)
test_data = []
for file in sorted(os.listdir("GuHanZi_dataset_bigsplit/testset")):
  if(not "jpg" in file):
    continue
  file = "GuHanZi_dataset_bigsplit/testset/" + file
  img = cv2.imread(file, 0)
  img = np.float32(img)/255
  if(model_type == "svc"):
    img = overlap_patches(img, 17, 6)
  test_data.append(img)
test_data = np.array(test_data)
f = open("GuHanZi_dataset_bigsplit/trainset/img_trainlabels.txt")
train_targets = np.array(list(map(int, f.read().split("\n"))))
f.close()
f = open("GuHanZi_dataset_bigsplit/testset/img_testlabels.txt")
test_targets = np.array(list(map(int, f.read().split("\n"))))
f.close()

char_trainset = CharDataset(
  data = train_data,
  targets = train_targets, transform = torch.from_numpy
)
char_testset = CharDataset(
  data = test_data,
  targets = test_targets, transform = torch.from_numpy
)
torch.manual_seed(886699)
data_loader_train = DataLoader(char_trainset, batch_size = 16, shuffle = True)
data_loader_test = DataLoader(char_testset, batch_size = 1, shuffle = False)

params = [p for p in vit.parameters() if(p.requires_grad)]
optimizer = torch.optim.Adam(params, lr = 0.0001, weight_decay = 0.01)
# optimizer = torch.optim.SGD(params, lr = 0.0001, momentum = 0.9, weight_decay = 0.01, nesterov = True)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.98)

print("train losses:")
vit.train()
train_losses = []
train_acc = 0
num_epochs = 30
for epoch in range(num_epochs):
  for imgs, labels in data_loader_train:
    imgs = imgs.to(device)
    labels = labels.to(device)
    preds = vit(imgs)
    train_loss = nn.CrossEntropyLoss()(preds, labels)
    print(train_loss)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    if(epoch >= num_epochs - 4):
      for i in range(len(preds)):
        train_acc += (preds[i, labels[i].item()] == max(preds[i]))
      train_losses.append(train_loss.to("cpu").item())
  lr_scheduler.step()
train_acc = train_acc*100/len(train_losses)/len(preds)

print("test losses:")
vit.eval()
test_losses = []
test_acc = 0
for imgs, labels in data_loader_test:
  imgs = imgs.to(device)
  labels = labels.to(device)
  preds = vit(imgs)
  test_acc += (preds[0, labels.item()] == max(preds[0]))
  test_loss = nn.CrossEntropyLoss()(preds, labels)
  print(test_loss)
  test_losses.append(test_loss.to("cpu").item())
test_acc = test_acc*100/len(test_losses)

print(f"average train loss:\n{sum(train_losses)/len(train_losses)}")
print(f"train accuracy:\n{train_acc}%")
print(f"average test loss:\n{sum(test_losses)/len(test_losses)}")
print(f"test accuracy:\n{test_acc}%")
