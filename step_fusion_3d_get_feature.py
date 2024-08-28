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
import pickle
import argparse

def get_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for i, (video_tensor, label) in enumerate(loader):
            video_tensor = video_tensor.to(device)
            output = model.get_vid_feature(video_tensor)
            features.append(output.cpu().numpy())
            
            if (i + 1) % 5 == 0:
                logging.info(f'Processed {i + 1} batches')
    return np.concatenate(features)




if __name__ == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser(description='Train 3d Video Swin Transformer')
    parser.add_argument('--batch_size', type=int, default=80,
                        help='input batch size for training (default: 80)')
    parser.add_argument('--seed', type=int, default=93728645,
                        help='random seed (default: 93728645)')
    
    # Model
    parser.add_argument('--path_3d_model', type=str, default='./saved_models/step_3d/final_3d.pth',
                        help='Path to pre-trained swin 3d model (default: ./saved_models/step_3d/final_3d.pth')
    parser.add_argument('--num_frames', type=int, default=77,
                        help='number of frames (time points)')
    parser.add_argument('--height', type=int, default=224,
                        help='frame height')
    parser.add_argument('--width', type=int, default=224,
                        help='frame width')

    # Inputs
    parser.add_argument('--path_video_train', type=str, default='./data/video_train.pkl',
                        help='Path to the video training data (default: ./data/video_train.pkl')
    parser.add_argument('--path_video_val', type=str, default='./data/video_val.pkl',
                        help='Path to the video validation data (default: ./data/video_val.pkl')
    
    parser.add_argument('--path_label_train', type=str, default='./data/label_train.pkl',
                        help='Path to training label (default: ./data/label_train.pkl')
    parser.add_argument('--path_label_val', type=str, default='./data/label_val.pkl',
                        help='Path to validation label (default: ./data/label_val.pkl')

    
    # Outputs
    parser.add_argument('--outpath', type=str, default='./video_features/',
                        help='where to save features')
    parser.add_argument('--logpath', type=str, default='./logs/',
                        help='where to save logs')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outpath = args.outpath
    
    
    # Configure logging
    logging.basicConfig(filename=args.logpath+'get_3d_features.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    logging.info('----- Define Data Loader -----')
    print('----- Define Data Loader -----')

    bsize=args.batch_size
    logging.info(f'batch size {bsize}')
    print(f'batch size {bsize}')
    
    logging.info(f'----- data path {args.path_video_train}')
    print(f'----- data path {args.path_video_train}')
  
    video_train = torch.from_numpy(pd.read_pickle(args.path_video_train).astype('float32'))
    video_val = torch.from_numpy(pd.read_pickle(args.path_video_val).astype('float32'))
    
    label_train = torch.from_numpy(pd.read_pickle(args.path_label_train))
    label_val = torch.from_numpy(pd.read_pickle(args.path_label_val))
    
    train_dataset = TensorDataset(video_train, label_train)
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)

    val_dataset = TensorDataset(video_val, label_val)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)

    
    logging.info('---- Define 3D swin ----')
    print('---- Define 3D swin ----')

    model = SwinTransformer3D(num_classes=4,
                              embed_dim = 128,
                               mlp_ratio=4.0,
                               patch_norm=True,
                               patch_size=(2,4,4,),
                               drop_path_rate=0.1, # stochastic depth rate in the paper
                               num_heads=[4,8,16,32],
                               depths = [2,2,18,2],
                               pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_base_patch4_window7_224.pth',
                               pretrained2d=True,
                               window_size=(8,7,7,))


    model.load_state_dict(torch.load(args.path_3d_model))
    
    model = model.to(device)
    
    logging.info('train loader')
    print('train loader')

    train_features = get_features(model, train_loader, device)
    with open(args.outpath+'train_3d_features.pkl', 'wb') as f:
        pickle.dump(train_features, f)
    logging.info('saved train')
    print('saved train')

    logging.info('val loader')
    print('val loader')

    val_features = get_features(model, val_loader, device)
    with open(args.outpath+'val_3d_features.pkl', 'wb') as f:
        pickle.dump(val_features, f)
    logging.info('saved val')
    print('saved val')

    



