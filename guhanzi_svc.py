import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
# from vit_pytorch import SimpleViT
from simple_vit_conv import SimpleViTConv
import os

device = torch.device("cuda") if(torch.cuda.is_available()) else torch.device("cpu")

# vit = SimpleViT(
vit = SimpleViTConv(
  image_size = 50,
  patch_size = 10,
  num_classes = 4,
  conv_dim = 8,
  dim = 1024,
  depth = 6,
  heads = 16,
  mlp_dim = 2048,
  channels = 1,
)

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
  train_data.append(img)
train_data = np.array(train_data)
test_data = []
for file in sorted(os.listdir("GuHanZi_dataset_bigsplit/testset")):
  if(not "jpg" in file):
    continue
  file = "GuHanZi_dataset_bigsplit/testset/" + file
  img = cv2.imread(file, 0)
  img = np.float32(img)/255
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
optimizer = torch.optim.Adam(params, lr = 0.0005, weight_decay = 0.01)
# optimizer = torch.optim.SGD(params, lr = 0.0005, momentum = 0.9, weight_decay = 0.01, nesterov = True)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.99)

print("train losses:")
vit.train()
train_losses = []
acc = 0
for epoch in range(25):
  for imgs, labels in data_loader_train:
    imgs = imgs.to(device)
    labels = labels.to(device)
    preds = vit(imgs)
    train_loss = torch.nn.CrossEntropyLoss()(preds, labels)
    print(train_loss)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    if(epoch > 20):
      for i in range(len(preds)):
        acc += (preds[i, labels[i].item()] == max(preds[i]))
      train_losses.append(train_loss.to("cpu").item())
  lr_scheduler.step()
print("average train loss:")
print(sum(train_losses)/len(train_losses))
print("train accuracy:")
print(f"{acc*100/len(train_losses)/len(preds)}%")

print("test losses:")
vit.eval()
test_losses = []
acc = 0
for imgs, labels in data_loader_test:
  imgs = imgs.to(device)
  labels = labels.to(device)
  preds = vit(imgs)
  acc += (preds[0, labels.item()] == max(preds[0]))
  test_loss = torch.nn.CrossEntropyLoss()(preds, labels)
  print(test_loss)
  test_losses.append(test_loss.to("cpu").item())
print("average test loss:")
print(sum(test_losses)/len(test_losses))
print("test accuracy:")
print(f"{acc*100/len(test_losses)}%")
