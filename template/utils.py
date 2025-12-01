import cv2
import torch
import random
import numpy as np
from .config import CFG
from .model import BiomassModel
from albumentations import (
    Compose, Resize, HorizontalFlip, VerticalFlip, RandomRotate90,
    ShiftScaleRotate, RandomBrightnessContrast, HueSaturationValue,
    RandomResizedCrop, CoarseDropout, Normalize
)
from albumentations.pytorch import ToTensorV2

def set_seed(seed=42, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(CFG.SEED, CFG.DETERMINISTIC)

# -------------------------
# 2) Augmentations
# -------------------------
def get_train_tf(img_size, aug_strength=1.0):
    """
    Get training augmentations with adjustable strength.
    
    Args:
        img_size: Image size
        aug_strength: Augmentation strength multiplier (1.0 = default, 0.0 = no augmentation, >1.0 = stronger)
    """
    # Scale augmentation parameters by strength
    shift_limit = 0.02 * aug_strength
    scale_limit = 0.1 * aug_strength
    rotate_limit = int(10 * aug_strength)
    hue_shift = int(10 * aug_strength)
    sat_shift = int(10 * aug_strength)
    val_shift = int(10 * aug_strength)
    brightness_limit = 0.15 * aug_strength
    contrast_limit = 0.15 * aug_strength
    dropout_p = min(0.3 * aug_strength, 1.0)
    
    return Compose([
        RandomResizedCrop(size=(img_size, img_size), scale=(0.85, 1.0), ratio=(0.95, 1.05), p=1.0),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.2),
        RandomRotate90(p=0.2),
        ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit, 
                        border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        HueSaturationValue(hue_shift_limit=hue_shift, sat_shift_limit=sat_shift, val_shift_limit=val_shift, p=0.3),
        RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=0.3),
        CoarseDropout(max_holes=4, max_height=int(img_size*0.08), max_width=int(img_size*0.08),
                      min_holes=1, fill_value=0, p=dropout_p),
        Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ], additional_targets={'image_right': 'image'} if CFG.DUAL_STREAM else {})

def get_valid_tf(img_size):
    return Compose([
        Resize(img_size, img_size),
        Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ], additional_targets={'image_right': 'image'} if CFG.DUAL_STREAM else {})

def kfold_split(df, n_folds=5, seed=42):
    from sklearn.model_selection import KFold
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df['fold'] = -1
    for i, (_, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, 'fold'] = i
    return df

def save_checkpoint(model, path):
    sd = model.state_dict()
    torch.save(sd, path)

def load_model(model_path, model_name=None, target_names=None, dual_stream=None, dropout=0.3, device=None):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint (.pth file)
        model_name: Model name (default: CFG.MODEL_NAME)
        target_names: Target names (default: CFG.TARGETS)
        dual_stream: Whether to use dual stream (default: CFG.DUAL_STREAM)
        dropout: Dropout rate (default: 0.3)
        device: Device to load model on (default: CFG.DEVICE)
    
    Returns:
        Loaded model in eval mode
    """
    if model_name is None:
        model_name = CFG.MODEL_NAME
    if target_names is None:
        target_names = CFG.TARGETS
    if dual_stream is None:
        dual_stream = CFG.DUAL_STREAM
    if device is None:
        device = CFG.DEVICE
    
    model = BiomassModel(
        model_name=model_name,
        pretrained=False,
        target_names=target_names,
        dual_stream=dual_stream,
        dropout=dropout
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model