import torch
import torch.nn as nn
from models.video_swin_transformer import SwinTransformer3D
from torch.utils.data import Dataset, DataLoader, TensorDataset

import cv2
import numpy as np
import pandas as pd
import random

import torch.optim as optim
from torch.nn import CrossEntropyLoss

import logging

from torch.optim.lr_scheduler import _LRScheduler
import math
import argparse

class CombinedLRScheduler(_LRScheduler):
    def __init__(self, optimizer, num_epochs, end_of_linear=2.5, eta_min=0, last_epoch=-1):
        self.num_epochs = num_epochs
        self.end_of_linear = end_of_linear
        self.eta_min = eta_min
        super(CombinedLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.end_of_linear:
            # Linear decay
            factor = 1 - self.last_epoch / self.end_of_linear
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            T_max = self.num_epochs - self.end_of_linear
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * (self.last_epoch - self.end_of_linear) / T_max)) / 2
                    for base_lr in self.base_lrs]


def evaluate_test_loss(model, test_loader, criterion, epoch, outpath):
    model.eval()
    running_test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs_3d, labels in test_loader:
            inputs_3d, labels = inputs_3d.to(device), labels.to(device)

            outputs = model(inputs_3d)
            test_loss = criterion(outputs, labels)
            running_test_loss += test_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_test_loss = running_test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    torch.save(model.state_dict(), outpath+'e'+str(epoch)+'_'+str(np.round(test_accuracy,4))+'.pth')
    return average_test_loss, test_accuracy


    
if __name__ == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser(description='Train 3d Video Swin Transformer')
    parser.add_argument('--batch_size', type=int, default=9,
                        help='input batch size for training (default: 9)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--seed', type=int, default=93728645,
                        help='random seed (default: 93728645)')
    parser.add_argument('--print_every', type=int, default=3,
                        help='epoch interval for validation and model saving (e.g., 3 for every 3 epochs).')

    # Model
    parser.add_argument('--num_frames', type=int, default=77,
                        help='number of frames (time points)')
    parser.add_argument('--height', type=int, default=224,
                        help='frame height')
    parser.add_argument('--width', type=int, default=224,
                        help='frame width')

    # Inputs
    parser.add_argument('--path_video_train', type=str, default='/home/sliang/data/video_train.pkl',
                        help='Path to the video training data (default: ./data/video_train.pkl')
    parser.add_argument('--path_video_val', type=str, default='/home/sliang/data/video_val.pkl',
                        help='Path to the video validation data (default: ./data/video_val.pkl')
    
    parser.add_argument('--path_label_train', type=str, default='/home/sliang/data/label_train.pkl',
                        help='Path to training label (default: ./data/label_train.pkl')
    parser.add_argument('--path_label_val', type=str, default='/home/sliang/data/label_val.pkl',
                        help='Path to validation label (default: ./data/label_val.pkl')
    
    # Outputs
    parser.add_argument('--outpath', type=str, default='./saved_models/step_fusion_3d/',
                        help='where to save models')
    parser.add_argument('--logpath', type=str, default='./logs/',
                        help='where to save logs')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outpath = args.outpath
    
    
    # Configure logging
    logging.basicConfig(filename=args.logpath+'step_3d_final.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    
    logging.info('----- Define Data Loader -----')
    bsize=args.batch_size
    logging.info(f'batch size {bsize}')
    
    logging.info(f'----- data path {args.path_video_train}')
  
    video_train = torch.from_numpy(pd.read_pickle(args.path_video_train))
    video_val = torch.from_numpy(pd.read_pickle(args.path_video_val))
    label_train = torch.from_numpy(pd.read_pickle(args.path_label_train))
    label_val = torch.from_numpy(pd.read_pickle(args.path_label_val))
    
    train_dataset = TensorDataset(video_train, label_train)
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)

    val_dataset = TensorDataset(video_val, label_val)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)
    
    logging.info('----- Define 3D swin -----')
    model = SwinTransformer3D(drop_path_rate=0.1,
                               mlp_ratio=4.0,
                               patch_norm=True,
                               patch_size=(2,4,4,),
                               pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth',
                               pretrained2d=True,
                               window_size=(8,7,7,))
    
    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = args.num_epochs
    print_every = args.print_every
    lr = args.lr
    logging.info(f'learning rate = {lr}')
    
    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=0.02)
    scheduler = CombinedLRScheduler(optimizer, num_epochs=num_epochs, end_of_linear=2.5)
    
    
    logging.info('start training')
    test_loss_list = []
    test_acc_list = []
    epoch_list = []
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0

        for i, (inputs_3d, labels) in enumerate(train_loader, 0):
            if i % 50 ==0:
                logging.info('---- '+str(i)+' th -----')
            inputs_3d, labels = inputs_3d.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs_3d)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % print_every == 0:
            epoch_list.append(epoch+1)
            test_loss_item, test_acc_item = evaluate_test_loss(model, val_loader, criterion, epoch+1, outpath)
            logging.info(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val Acc: {test_acc_item}")
            test_loss_list.append(test_loss_item)
            test_acc_list.append(test_acc_item)


        scheduler.step()
    logging.info("Training completed!")
    
    torch.save(model.state_dict(), outpath+'final_3d.pth')
    np.save(outpath+'val_loss.npy', np.array(test_loss_list))
    np.save(outpath+'val_acc.npy', np.array(test_acc_list))
    logging.info(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')