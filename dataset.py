import os
import numpy as np
from tqdm import tqdm

import medmnist
from medmnist import INFO
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from collections import OrderedDict
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from utils import Transform3D, model_to_syncbn
from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def extract_data(data_flag, download, BATCH_SIZE):

    info = INFO[data_flag]
    task = info['task']

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    # data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[.5], std=[.5])
    # ])

    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=True)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=True)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=True)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    return train_loader, train_loader_at_eval, val_loader, test_loader

def extract_data_3d(data_flag, download, BATCH_SIZE):
    info = INFO[data_flag]

    DataClass = getattr(medmnist, info['python_class'])

    train_transform = Transform3D(mul='random') if True else Transform3D()
    eval_transform = Transform3D(mul='0.5') if True else Transform3D()

    train_dataset = DataClass(split='train', transform=train_transform, download=download, as_rgb=True)
    train_dataset_at_eval = DataClass(split='train', transform=eval_transform, download=download, as_rgb=True)
    val_dataset = DataClass(split='val', transform=eval_transform, download=download, as_rgb=True)
    test_dataset = DataClass(split='test', transform=eval_transform, download=download, as_rgb=True)

    
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset_at_eval, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('==> Building and training model...')
    return train_loader, train_loader_at_eval, val_loader, test_loader

def train(model, train_loader, task, criterion, optimizer, device, writer):
    total_loss = []
    global iteration
    iteration = 0

    model.train()
    for inputs, targets in tqdm(train_loader, desc="train"):
    # for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = torch.squeeze(targets, 1).long().to(device)
        loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()
    
    epoch_loss = sum(total_loss)/len(total_loss)
    print("Epoch loss", epoch_loss)

    return epoch_loss

def test(model, evaluator, data_loader, task, criterion, device, run, save_folder=None):

    model.eval()
    
    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        # for inputs, targets in tqdm(data_loader, desc="test"):
        for inputs, targets in data_loader:
            outputs = model(inputs.to(device))
            
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)
        
        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]

def train_2d_model(model, DEVICE, data_flag, milestones, gamma, output_root, task, train_loader, train_loader_at_eval, val_loader, test_loader, NUM_EPOCHS, lr, hyperparameters):
    
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    train_evaluator = medmnist.Evaluator(data_flag, 'train')
    val_evaluator = medmnist.Evaluator(data_flag, 'val')
    test_evaluator = medmnist.Evaluator(data_flag, 'test')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)

    writer = SummaryWriter(log_dir=os.path.join(output_root, f'{model.name}_{hyperparameters}'))

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)

    for epoch in range(NUM_EPOCHS): 
        print(f"Epoch {epoch} of {NUM_EPOCHS - 1}")
        train_loss = train(model, train_loader, task, criterion, optimizer, DEVICE, writer)
        
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, DEVICE, model.name)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, DEVICE, model.name)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, DEVICE, model.name)
        
        scheduler.step()
        
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)
            
        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    state = best_model.state_dict()
    print(state)
    path = os.path.join(output_root, f'{model.name}_best_model_{hyperparameters}.pth')
    torch.save(state, path)

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, task, criterion, DEVICE, model.name, output_root)
    val_metrics = test(best_model, val_evaluator, val_loader, task, criterion, DEVICE, model.name, output_root)
    test_metrics = test(best_model, test_evaluator, test_loader, task, criterion, DEVICE, model.name, output_root)

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log
    print(log)
            
    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)  

    writer.close()

def train_3d_model(model, DEVICE, data_flag, milestones, gamma, output_root, task, train_loader, train_loader_at_eval, val_loader, test_loader, NUM_EPOCHS, lr, hyperparameters):
    conv = 'Conv3d'
    pretrained_3d = 'i3d'
    model_flag = 'resnet18'

    if conv=='ACSConv':
        model = model_to_syncbn(ACSConverter(model))
    if conv=='Conv2_5d':
        model = model_to_syncbn(Conv2_5dConverter(model))
    if conv=='Conv3d':
        if pretrained_3d == 'i3d':
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
        else:
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    train_evaluator = medmnist.Evaluator(data_flag, 'train')
    val_evaluator = medmnist.Evaluator(data_flag, 'val')
    test_evaluator = medmnist.Evaluator(data_flag, 'test')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)

    writer = SummaryWriter(log_dir=os.path.join(output_root, f'{model.name}_{hyperparameters}'))

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)

    for epoch in range(NUM_EPOCHS):  
        print(f"Epoch {epoch} of {NUM_EPOCHS - 1}")      
        train_loss = train(model, train_loader, task, criterion, optimizer, DEVICE, writer)
        
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, DEVICE, model.name)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, DEVICE, model.name)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, DEVICE, model.name)
        
        scheduler.step()
        
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)
            
        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    state = best_model.state_dict()

    path = os.path.join(output_root, f'{model.name}_best_model_{hyperparameters}.pth')
    torch.save(state, path)

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, task, criterion, DEVICE, model.name, output_root)
    val_metrics = test(best_model, val_evaluator, val_loader, task, criterion, DEVICE, model.name, output_root)
    test_metrics = test(best_model, test_evaluator, test_loader, task, criterion, DEVICE, model.name, output_root)

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log
    print(log)
            
    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)  

    writer.close()

def plot_confusion_matrix(model, model_type, test_loader, hyperparameters, output_root):
    model.load_state_dict(torch.load(f'{output_root}/{model_type}_best_model_{hyperparameters}.pth'))
    model.eval()

    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)

            all_true_labels.extend(torch.squeeze(targets, 1).long())
            all_pred_labels.extend(predicted.cpu().numpy())

    y_true = np.array(all_true_labels)
    y_pred = np.array(all_pred_labels)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f'{output_root}/{model_type}_best_model_{hyperparameters}.png')
    # plt.show()