#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score

from utils import train, evaluate
from plots import plot_learning_curves, plot_confusion_matrix


# In[ ]:


cuda = torch.cuda.is_available()
torch.manual_seed(6250)
if cuda:
    torch.cuda.manual_seed(6250)


# In[ ]:


kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

# CheXpert specific metadata
n_classes = 14
im_size = (3, 224, 224)

im_stat = pd.read_csv("../CheXpert-v1.0-small/image_statistics.csv", names = ["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B"])
# Subtract the mean color and divide by standard deviation.
chexpert_mean_color = [im_stat["mean_R"][0], im_stat["mean_G"][0], im_stat["mean_B"][0]] 
# std dev of color across training images
chexpert_std_color = [im_stat["std_R"][0], im_stat["std_G"][0], im_stat["std_B"][0]]

#normalization if using pretrained
pretrained_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


# In[ ]:


class CheXpertDataset(Dataset):
    def __init__(self, split, path="../CheXpert-v1.0-small/", transform=None):       
        dataset = pd.read_csv(path + split + ".csv")

        self.image_paths = dataset.Path.apply(lambda x: "../"+x).tolist()
        self.labels = dataset.iloc[:,5:].values
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = torch.from_numpy(self.labels[index].astype('float32')).float()
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)


# In[ ]:


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class ResNet152(nn.Module):
    def __init__(self, output_size, pretrained=False):
        super(ResNet152, self).__init__()
        self.resnet152 = torchvision.models.resnet152(pretrained = pretrained)
        set_parameter_requires_grad(self.resnet152, pretrained)
        resnet_features = self.resnet152.fc.in_features
        # re-implement final layer as output_size is different
        self.resnet152.fc = nn.Linear(resnet_features, output_size)
        #normalization to get 0-1 scores
        self.finalSigmoid = nn.Sigmoid() 

    def forward(self, input):
        x = self.resnet152(input)
        x = self.finalSigmoid(x)
        return x

class DenseNet161(nn.Module):
    def __init__(self, output_size, pretrained=False):
        super(DenseNet161, self).__init__()
        self.densenet161 = torchvision.models.densenet161(pretrained = pretrained)
        set_parameter_requires_grad(self.densenet161, pretrained)
        resnet_features = self.densenet161.classifier.in_features
        # re-implement final layer as output_size is different
        self.densenet161.classifier = nn.Linear(resnet_features, output_size)
        #normalization to get 0-1 scores
        self.finalSigmoid = nn.Sigmoid() 

    def forward(self, input):
        x = self.densenet161(input)
        x = self.finalSigmoid(x)
        return x


# In[ ]:


NUM_EPOCHS = 10
BATCH_SIZE = 64
lr = 0.001
weight_decay = 0
device = torch.device("cuda" if cuda else "cpu")
pretrained = False

model = ResNet152(n_classes, pretrained)
criterion = F.binary_cross_entropy


# In[ ]:

#transforms
transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                pretrained_normalize if pretrained else transforms.Normalize(chexpert_mean_color, chexpert_std_color),
            ])
transform_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),  
                transforms.RandomApply([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                    transforms.RandomResizedCrop(224, scale=(0.85, 1.0))
                ]),
                transforms.ToTensor(),
                pretrained_normalize if pretrained else transforms.Normalize(chexpert_mean_color, chexpert_std_color),
            ])

# Datasets
train_dataset = CheXpertDataset(split='train_train', transform=transform_train)
val_dataset = CheXpertDataset(split='train_valid', transform=transform)
test_dataset = CheXpertDataset(split='valid', transform=transform)
# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(val_dataset,
                batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,
                batch_size=BATCH_SIZE, shuffle=True, **kwargs)


# In[ ]:


if cuda:
    model.cuda()

params_to_update = model.parameters()
if pretrained:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
optimizer = optim.Adam(params_to_update, lr = lr, weight_decay = weight_decay)


# In[ ]:


best_mean_auc = 0.0
train_losses, train_auc_lbs = [], []
valid_losses, valid_auc_lbs = [], []

for epoch in range(NUM_EPOCHS):
   train_loss, train_auc_lb = train(model, device, train_loader, criterion, optimizer, epoch)
   valid_loss, valid_auc_lb, valid_results = evaluate(model, device, valid_loader, criterion)

   train_losses.append(train_loss)
   valid_losses.append(valid_loss)

   train_auc_lbs.append(train_auc_lb)
   valid_auc_lbs.append(valid_auc_lb)
    
#    train_auc_ubs.append(train_auc_ub)
#    valid_auc_ubs.append(valid_auc_ub)

   is_best = np.mean((valid_auc_lb)) > best_mean_auc  # let's keep the model that has the best avg. AUC, but you can also use another metric.
   if is_best:
       best_mean_auc = np.mean((valid_auc_lb))
       torch.save(model, "CheXpert-model.pt")

plot_learning_curves(train_losses, valid_losses, train_auc_lbs, valid_auc_lbs)

best_model = torch.load("CheXpert-model.pt")
test_loss, test_auc_lb, test_results = evaluate(best_model, device, test_loader, criterion, print_freq=1)
r = np.array(test_results)
pc_f1 = [f1_score(r[:,0,label], r[:,1,label]) for label in np.arange(r.shape[2])]
print("Per-class F1: ", np.around(pc_f1, 3))
print("Avg. F1: ", np.around(np.mean(pc_f1, axis = 0), 3))

class_names = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
plot_confusion_matrix(test_results, class_names)

