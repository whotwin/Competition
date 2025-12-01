import os, gc
from utils import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import TestDataset
from train import train_one_fold
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

FIVE_TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
ALL_TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'GDM_g', 'Dry_Total_g']
INDEX_COLS = ['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']

def _dup_check_for_pivot(df_long, index_cols=INDEX_COLS, name_col='target_name'):
    keys = index_cols + [name_col]
    dup_mask = df_long.duplicated(keys, keep=False)
    return df_long.loc[dup_mask, keys].value_counts().reset_index(name='count')

def long_to_wide_for_training(
    df_long: pd.DataFrame,
    targets=('Dry_Total_g','GDM_g','Dry_Green_g'),
    strict=True,
    aggfunc='first'
) -> pd.DataFrame:
    """
    長形式(train.csv) → 学習用の広形式（画像1行＋3ターゲット列）。
    余剰の2ターゲット(Dry_Dead/Clover)は保持してもOKだが、学習では未使用。
    """
    # 1) 重複チェック（必要なら平均等で潰す）
    if strict:
        dups = _dup_check_for_pivot(df_long)
        if len(dups):
            raise ValueError(
                f"Pivot keys have duplicates ({len(dups)} rows). "
                f"Set strict=False or aggfunc='mean'.\n{dups.head()}"
            )

    # 2) ピボット
    wide = df_long.pivot_table(
        index=INDEX_COLS,
        columns='target_name',
        values='target',
        aggfunc=aggfunc
    ).reset_index()

    # 3) 学習で使う3列が無ければエラー
    for t in targets:
        if t not in wide.columns:
            raise KeyError(f"Required target column missing after pivot: {t}")

    # 4) image_id 付与（任意）
    wide['image_id'] = wide['image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    keep_cols = list(INDEX_COLS) + list(targets) + ['image_id']
    keep_cols = [c for c in keep_cols if c in wide.columns]
    wide = wide[keep_cols].copy()
    return wide

def add_stratified_folds(
    df: pd.DataFrame,
    n_folds=5,
    label_col='Dry_Total_g',
    bins=5,
    seed=42
) -> pd.DataFrame:
    """
    Dry_Total_g をビニングして層化KFold。ターゲット分布を各foldで均等化しやすい。
    """
    df = df.copy()
    # 欠損・定数対応（qcutが失敗しないように）
    y = df[label_col].values
    # 一意値が少なければ bin 数を調整
    uniq = np.unique(y)
    bins = min(bins, max(2, len(uniq)))
    # 量子化。重複binに配慮して duplicates='drop'
    df['_strat'] = pd.qcut(y, q=bins, labels=False, duplicates='drop')

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df['fold'] = -1
    for f, (_, val_idx) in enumerate(skf.split(df, df['_strat'])):
        df.loc[val_idx, 'fold'] = f
    df = df.drop(columns=['_strat'])
    return df

def save_training_csv_for_existing_pipeline(df_wide: pd.DataFrame, out_path: str):
    cols_needed = ['image_path', 'Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    missing = [c for c in cols_needed if c not in df_wide.columns]
    if missing:
        raise KeyError(f"Columns missing for training: {missing}")
    df_wide.to_csv(out_path, index=False)
    print(f"Saved training CSV for pipeline: {out_path}  shape={df_wide.shape}")

def predict_test_set(
    model,
    test_df,
    image_dir,
    transform,
    batch_size=32,
    dual_stream=None,
    device=None,
    num_workers=2,
    tta=False
):
    """
    Predict on entire test set.
    
    Args:
        model: Trained model
        test_df: DataFrame with 'image_path' column
        image_dir: Directory containing test images
        transform: Transform to apply
        batch_size: Batch size for inference
        dual_stream: Whether to use dual stream (default: CFG.DUAL_STREAM)
        device: Device to use (default: CFG.DEVICE)
        num_workers: Number of workers for DataLoader
        tta: Whether to use test-time augmentation
    
    Returns:
        numpy array of predictions [N, num_targets]
    """
    if dual_stream is None:
        dual_stream = CFG.DUAL_STREAM
    if device is None:
        device = CFG.DEVICE
    
    dataset = TestDataset(test_df, image_dir, transform, dual_stream=dual_stream)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    
    all_predictions = []
    model.eval()
    
    with torch.no_grad():
        for left, right, indices in tqdm(dataloader, desc="Predicting"):
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
            
            if tta:
                # TTA: original, hflip, vflip, hvflip
                preds_tta = []
                for hflip, vflip in [(False, False), (True, False), (False, True), (True, True)]:
                    left_aug = torch.flip(left, dims=[3] if hflip else []) if hflip else left
                    left_aug = torch.flip(left_aug, dims=[2] if vflip else []) if vflip else left_aug
                    right_aug = torch.flip(right, dims=[3] if hflip else []) if hflip else right
                    right_aug = torch.flip(right_aug, dims=[2] if vflip else []) if vflip else right_aug
                    
                    with torch.cuda.amp.autocast(enabled=CFG.USE_AMP):
                        pred = model(left_aug, right_aug)  # [B, num_targets]
                    preds_tta.append(pred)
                
                pred = torch.stack(preds_tta, dim=0).mean(dim=0)  # Average over TTA
            else:
                with torch.cuda.amp.autocast(enabled=CFG.USE_AMP):
                    pred = model(left, right)  # [B, num_targets]
            
            all_predictions.append(pred.cpu().numpy())
    
    return np.concatenate(all_predictions, axis=0)

def ensemble_predict(
    test_df,
    model_paths,
    image_dir,
    model_name=None,
    target_names=None,
    dual_stream=None,
    dropout=0.3,
    batch_size=32,
    device=None,
    num_workers=2,
    tta=False,
    weights=None
):
    """
    Ensemble predictions from multiple models (typically different folds).
    
    Args:
        test_df: DataFrame with 'image_path' column
        model_paths: List of paths to model checkpoints
        image_dir: Directory containing test images
        model_name: Model name (default: CFG.MODEL_NAME)
        target_names: Target names (default: CFG.TARGETS)
        dual_stream: Whether to use dual stream (default: CFG.DUAL_STREAM)
        dropout: Dropout rate (default: 0.3)
        batch_size: Batch size for inference
        device: Device to use (default: CFG.DEVICE)
        num_workers: Number of workers for DataLoader
        tta: Whether to use test-time augmentation
        weights: Weights for each model (default: equal weights)
    
    Returns:
        numpy array of ensemble predictions [N, num_targets]
    """
    if model_name is None:
        model_name = CFG.MODEL_NAME
    if target_names is None:
        target_names = CFG.TARGETS
    if dual_stream is None:
        dual_stream = CFG.DUAL_STREAM
    if device is None:
        device = CFG.DEVICE
    if weights is None:
        weights = [1.0] * len(model_paths)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    transform = get_valid_tf(CFG.IMG_SIZE)
    
    all_predictions = []
    
    for i, model_path in enumerate(model_paths):
        print(f"Loading model {i+1}/{len(model_paths)}: {model_path}")
        model = load_model(
            model_path,
            model_name=model_name,
            target_names=target_names,
            dual_stream=dual_stream,
            dropout=dropout,
            device=device
        )
        
        pred = predict_test_set(
            model=model,
            test_df=test_df,
            image_dir=image_dir,
            transform=transform,
            batch_size=batch_size,
            dual_stream=dual_stream,
            device=device,
            num_workers=num_workers,
            tta=tta
        )
        
        all_predictions.append(pred * weights[i])
        
        # Clean up
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    ensemble_pred = np.sum(all_predictions, axis=0)
    return ensemble_pred

def predict_and_save_submission(
    test_csv_path,
    output_path,
    model_paths,
    image_dir=None,
    model_name=None,
    target_names=None,
    dual_stream=None,
    dropout=0.3,
    batch_size=None,
    device=None,
    num_workers=None,
    tta=None,
    weights=None,
    use_log1p=None
):
    """
    Predict on test set and save submission file in the required format.
    
    Args:
        test_csv_path: Path to test.csv (long format)
        output_path: Path to save submission CSV
        model_paths: List of paths to model checkpoints (or single path)
        image_dir: Directory containing test images (default: inferred from BASE_PATH)
        model_name: Model name (default: CFG.MODEL_NAME)
        target_names: Target names (default: CFG.TARGETS)
        dual_stream: Whether to use dual stream (default: CFG.DUAL_STREAM)
        dropout: Dropout rate (default: 0.3)
        batch_size: Batch size for inference
        device: Device to use (default: CFG.DEVICE)
        num_workers: Number of workers for DataLoader
        tta: Whether to use test-time augmentation
        weights: Weights for each model (default: equal weights)
        use_log1p: Whether model was trained with log1p (default: CFG.USE_LOG1P)
    
    Returns:
        DataFrame with predictions
    """
    if image_dir is None:
        image_dir = CFG.TEST_IMAGE_DIR
    if model_name is None:
        model_name = CFG.MODEL_NAME
    if target_names is None:
        target_names = CFG.TARGETS
    if dual_stream is None:
        dual_stream = CFG.DUAL_STREAM
    if device is None:
        device = CFG.DEVICE
    if use_log1p is None:
        use_log1p = CFG.USE_LOG1P
    if batch_size is None:
        batch_size = CFG.INFERENCE_BATCH_SIZE
    if num_workers is None:
        num_workers = CFG.NUM_WORKERS
    if tta is None:
        tta = CFG.USE_TTA
    
    # Load test CSV (long format)
    test_df_long = pd.read_csv(test_csv_path)
    
    # Get unique image paths
    unique_images = test_df_long['image_path'].unique()
    test_df = pd.DataFrame({'image_path': unique_images})
    
    # Convert model_paths to list if single path
    if isinstance(model_paths, str):
        model_paths = [model_paths]
    
    # Get predictions
    print(f"Predicting on {len(test_df)} images using {len(model_paths)} model(s)...")
    predictions = ensemble_predict(
        test_df=test_df,
        model_paths=model_paths,
        image_dir=image_dir,
        model_name=model_name,
        target_names=target_names,
        dual_stream=dual_stream,
        dropout=dropout,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        tta=tta,
        weights=weights
    )
    
    # Convert log1p predictions back to original scale
    if use_log1p:
        predictions = np.expm1(predictions)
    
    # Create prediction DataFrame
    pred_df = pd.DataFrame(
        predictions,
        columns=target_names,
        index=test_df.index
    )
    pred_df['image_path'] = test_df['image_path'].values
    
    # Convert to long format for submission
    submission_rows = []
    for _, row in test_df_long.iterrows():
        image_path = row['image_path']
        target_name = row['target_name']
        
        # Find corresponding prediction
        pred_row = pred_df[pred_df['image_path'] == image_path].iloc[0]
        
        # Get the 3 main targets
        total = pred_row['Dry_Total_g']
        gdm = pred_row['GDM_g']
        green = pred_row['Dry_Green_g']
        
        # Derive all 5 targets
        if target_name == 'Dry_Total_g':
            value = total
        elif target_name == 'GDM_g':
            value = gdm
        elif target_name == 'Dry_Green_g':
            value = green
        elif target_name == 'Dry_Dead_g':
            value = total - gdm
        elif target_name == 'Dry_Clover_g':
            value = gdm - green
        else:
            value = 0.0
        
        submission_rows.append({
            'sample_id': row['sample_id'],
            'target': max(0.0, value)  # Ensure non-negative
        })
    
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    
    return submission_df

def main(lr=None, batch_size=None, wd=None, warmup_epochs=None, dropout=None, aug_strength=None):
    """
    Main training/inference function with optional parameter overrides.
    
    Args:
        lr: Learning rate override (default: CFG.LR)
        batch_size: Batch size override (default: CFG.BATCH_SIZE)
        wd: Weight decay override (default: CFG.WD)
        warmup_epochs: Warmup epochs override (default: CFG.WARMUP_EPOCHS)
        dropout: Dropout rate override (default: 0.3)
        aug_strength: Augmentation strength override (default: 1.0)
    
    Returns:
        dict: Results containing 'cv_mean', 'cv_std', and 'fold_scores' (training mode)
              or DataFrame with predictions (inference mode)
    """
    if CFG.INFERENCE_MODE:
        print("=" * 50)
        print("INFERENCE MODE")
        print("=" * 50)

        model_dir = CFG.INFERENCE_MODEL_DIR if CFG.INFERENCE_MODEL_DIR is not None else CFG.OUT_DIR
        print(f"Loading models from: {model_dir}")
        
        # 使用するfoldを決定
        if CFG.INFERENCE_FOLDS is None:
            # 全foldを自動検出
            model_paths = []
            for fold in range(CFG.N_FOLDS):
                model_path = os.path.join(model_dir, f'best_model_fold{fold}.pth')
                if os.path.exists(model_path):
                    model_paths.append(model_path)
                else:
                    print(f"Warning: Model not found: {model_path}")
            
            if len(model_paths) == 0:
                raise ValueError(f"No model files found in {model_dir}")
            
            print(f"Found {len(model_paths)} model(s): {[os.path.basename(p) for p in model_paths]}")
        else:
            # 指定されたfoldのみ使用
            model_paths = []
            for fold in CFG.INFERENCE_FOLDS:
                model_path = os.path.join(model_dir, f'best_model_fold{fold}.pth')
                if os.path.exists(model_path):
                    model_paths.append(model_path)
                else:
                    raise FileNotFoundError(f"Model not found: {model_path}")
            print(f"Using {len(model_paths)} model(s) from folds: {CFG.INFERENCE_FOLDS}")
        
        # 推論実行
        submission_df = predict_and_save_submission(
            test_csv_path=CFG.TEST_CSV,
            output_path=CFG.SUBMISSION_OUTPUT,
            model_paths=model_paths,
            image_dir=CFG.TEST_IMAGE_DIR,
            model_name=CFG.MODEL_NAME,
            target_names=CFG.TARGETS,
            dual_stream=CFG.DUAL_STREAM,
            dropout=dropout if dropout is not None else 0.3,
            batch_size=CFG.INFERENCE_BATCH_SIZE,
            device=CFG.DEVICE,
            num_workers=CFG.NUM_WORKERS,
            tta=CFG.USE_TTA,
            weights=None,
            use_log1p=CFG.USE_LOG1P
        )
        
        print("=" * 50)
        print(f"Inference completed. Submission saved to: {CFG.SUBMISSION_OUTPUT}")
        print("=" * 50)
        
        return submission_df
    
    # 学習モード（既存の処理）
    print("=" * 50)
    print("TRAINING MODE")
    print("=" * 50)
    df = pd.read_csv(CFG.TRAIN_CSV)

    target_list = FIVE_TARGET_ORDER if CFG.TRAIN_FIVE_OUTPUT_LOSS else ['Dry_Total_g','GDM_g','Dry_Green_g']
    CFG.TARGETS = target_list
    df = long_to_wide_for_training(
        df,
        targets=tuple(target_list),
        strict=True,
        aggfunc='first'
    )

    df = add_stratified_folds(
        df,
        n_folds=5,
        label_col='Dry_Total_g',
        bins=5,
        seed=42
    )

    assert set(CFG.TARGETS).issubset(df.columns), f"train.csvに{CFG.TARGETS}が必要です"

    if 'fold' not in df.columns:
        df = kfold_split(df, n_folds=CFG.N_FOLDS, seed=CFG.SEED)

    df.to_csv(os.path.join(CFG.OUT_DIR, 'train_folds.csv'), index=False)
    print("Folds saved:", os.path.join(CFG.OUT_DIR, 'train_folds.csv'))

    bests = []
    for f in range(CFG.N_FOLDS):
        best_metric = train_one_fold(df, f, lr=lr, batch_size=batch_size, wd=wd, 
                                      warmup_epochs=warmup_epochs, dropout=dropout, aug_strength=aug_strength)
        bests.append(best_metric)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    metric_name = 'Weighted R2' if CFG.SELECT_BEST_BY.lower() == 'r2' else 'RMSE'
    cv_mean = np.mean(bests)
    cv_std = np.std(bests)
    print(f"\n=== CV {metric_name} (mean±std) ===")
    print(f"{cv_mean:.5f} ± {cv_std:.5f}")
    
    return {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'fold_scores': bests
    }

if __name__ == '__main__':
    main()