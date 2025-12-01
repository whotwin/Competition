import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def gwml_calculation(weight_cls, total_labels):
    # Default dtype np.float32
    # total_labels be like: 
    # --------------------------------------------------------------
    # Dry_Green_g | Dry_Dead_g | Dry_Clover_g | GDM_g | Dry_Total-G |
    #    1              2            3            4         10
    #    .              .            .            .         .
    #    .              .            .            .         .
    #    .              .            .            .         .
    # --------------------------------------------------------------
    weight_cls = weight_cls[None, :] * total_labels
    global_weighted_mean_label = np.mean(weight_cls, axis=1, keepdims=True) # [N, 1]
    return global_weighted_mean_label

def make_kfolds(image_list, groups, n_splits=5, seed=42):
    image_list = np.array(image_list)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(image_list)):
        train_imgs = image_list[train_idx].tolist()
        val_imgs = image_list[val_idx].tolist()
        folds.append({
            "train": train_imgs,
            "val": val_imgs
        })
        print(f"Fold {fold_idx}: train {len(train_imgs)} images, val {len(val_imgs)} images")
    return folds

def read_csv(train_csv_path):
    df = pd.read_csv(train_csv_path)
    if df.isnull().values.any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            f"CSV 文件中存在缺失值：{null_cols}"
        )

    # --- 每张图片对应 5 行，需要 groupby 合并 ---
    groups = {}  # { image_path : { target_name:value, ... } }

    for img, sub_df in df.groupby("image_path"):
        # sub_df 包含 5 行
        tdict = {row["target_name"]: row["target"] for _, row in sub_df.iterrows()}
        groups[img] = tdict

    # 保存 unique image list
    image_list = list(groups.keys())

    print(f"Loaded {len(image_list)} unique images.")
    return image_list, groups, df