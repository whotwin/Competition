import os
from tqdm import tqdm
from dataset import CSIRO_Dataset
from torch.utils.data import DataLoader
from utils import read_csv, make_kfolds

def train_loop(dataloader, model):
    device = model.device()
    model.train()
    for i, data in enumerate(tqdm(dataloader)):
        image, targets = data
        image, targets = image.to(device), targets.to(device)
        preds = model(image)

def train_one_fold(fold, cfg):
    train_dataset = CSIRO_Dataset(cfg, fold['train'])
    val_dataset = CSIRO_Dataset(cfg, fold['val'])
    train_dataloader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workder)
    val_dataloader = DataLoader(val_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workder)


def train(cfg):
    image_list, groups, df = read_csv(cfg.train_csv_path)
    folds = make_kfolds(image_list, groups, n_splits=5, seed=42)
    dataset = CSIRO_Dataset(cfg)