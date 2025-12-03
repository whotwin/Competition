import os
from dataset import CSIRO_Dataset
from utils import read_csv, make_kfolds

def train(cfg):
    image_list, groups, df = read_csv(cfg.train_csv_path)
    folds = make_kfolds(image_list, groups, n_splits=5, seed=42)
    dataset = CSIRO_Dataset(cfg)