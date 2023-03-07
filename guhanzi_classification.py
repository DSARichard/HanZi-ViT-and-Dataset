import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import SimpleViT
from vit_pytorch.cct import CCT
import os

model_type = ["vit", "cnn", "cct"][0]
device = torch.device("cuda") if(torch.cuda.is_available()) else torch.device("cpu")

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 12, 3)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(12, 16, 3)
    self.fc1 = nn.Linear(1936, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 256)
    self.fc4 = nn.Linear(256, 4)
  
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x

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

cct = CCT(
    img_size = 50,
    embedding_dim = 192,
    n_input_channels = 1,
    n_conv_layers = 2,
    kernel_size = 3,
    stride = 1,
    padding = 1,
    pooling_kernel_size = 3,
    pooling_stride = 1,
    pooling_padding = 1,
    num_layers = 6,
    num_heads = 6,
    mlp_ratio = 2.0,
    num_classes = 4,
    positional_embedding = 'learnable',
)

vit = vit if(model_type == "vit") else CNN() if(model_type == "cnn") else cct
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
for file in sorted(os.listdir("GuHanZi_dataset/trainset")):
  if(not "jpg" in file):
    continue
  file = "GuHanZi_dataset/trainset/" + file
  img = cv2.imread(file, 0)
  img = np.float32(img)/255
  train_data.append(img)
train_data = np.array(train_data)
test_data = []
for file in sorted(os.listdir("GuHanZi_dataset/testset")):
  if(not "jpg" in file):
    continue
  file = "GuHanZi_dataset/testset/" + file
  img = cv2.imread(file, 0)
  img = np.float32(img)/255
  test_data.append(img)
test_data = np.array(test_data)
f = open("GuHanZi_dataset/trainset/img_trainlabels.txt")
train_targets = np.array(list(map(int, f.read().split("\n"))))
f.close()
f = open("GuHanZi_dataset/testset/img_testlabels.txt")
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
torch.manual_seed(447585)
data_loader_train = DataLoader(char_trainset, batch_size = 16 if(model_type == "cct") else 64, shuffle = True)
data_loader_test = DataLoader(char_testset, batch_size = 1, shuffle = False)

params = [p for p in vit.parameters() if(p.requires_grad)]
optimizer = torch.optim.Adam(params, lr = 0.0005, weight_decay = 0.01)
# optimizer = torch.optim.SGD(params, lr = 0.0005, momentum = 0.9, weight_decay = 0.01, nesterov = True)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.99)

print("train losses:")
vit.train()
train_losses = []
for epoch in range(20):
  for imgs, labels in data_loader_train:
    imgs = imgs.to(device)
    labels = labels.to(device)
    preds = vit(imgs)
    train_loss = torch.nn.CrossEntropyLoss()(preds, labels)
    print(train_loss)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.to("cpu").item())
  lr_scheduler.step()
print("average train loss:")
print(sum(train_losses)/len(train_losses))

print("test losses:")
vit.eval()
test_losses = []
acc = 0
for imgs, labels in data_loader_test:
  imgs = imgs.to(device)
  labels = labels.to(device)
  preds = vit(imgs)
  acc += (preds[0, labels.item()] == max(preds[0]))
  print(preds)
  print(labels)
  test_loss = torch.nn.CrossEntropyLoss()(preds, labels)
  print(test_loss)
  test_losses.append(test_loss.to("cpu").item())
print("average test loss:")
print(sum(test_losses)/len(test_losses))
print("accuracy:")
print(f"{acc*100/100}%")
