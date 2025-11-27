import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TARGETS = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
DUAL_STREAM = True

class TrainDataset(Dataset):

    def __init__(self, df, image_dir, tf, use_log1p=True):
        self.df = df.reset_index(drop=True)
        self.paths = self.df['image_path'].values
        self.y = self.df[TARGETS].values.astype(np.float32)
        self.image_dir = image_dir
        self.tf = tf
        self.use_log1p = use_log1p

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        raw_path = self.paths[idx]
        # Use provided path if it exists; otherwise, fall back to joining with image_dir and basename
        candidate = raw_path if os.path.exists(raw_path) else os.path.join(self.image_dir, os.path.basename(raw_path))
        img = cv2.imread(candidate)
        if img is None:
            img = np.zeros((1000,2000,3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if DUAL_STREAM:
            h, w, _ = img.shape
            mid = w//2
            left = img[:, :mid]
            right = img[:, mid:]
            t = self.tf(image=left, image_right=right)
            left = t['image']
            right = t['image_right']
        else:
            t = self.tf(image=img)
            left = t['image']
            right = left

        target = self.y[idx].copy()
        if self.use_log1p:
            target = np.log1p(target)
        target = torch.from_numpy(target)

        return left, right, target