import os
import math
from utils import *
from tqdm import tqdm
from torch.optim import AdamW
from dataset import TrainDataset
from torch.utils.data import DataLoader
from model import BiomassModel, ModelEMA, WeightedMSELoss, metric_rmse
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

CSIRO_WEIGHTS = {
    'Dry_Green_g': 0.10,
    'Dry_Dead_g':  0.10,
    'Dry_Clover_g':0.10,
    'GDM_g':       0.20,
    'Dry_Total_g': 0.50,
}

FIVE_TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def _r2_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    if y_true.size == 0: return np.nan
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - y_true.mean()) ** 2)
    if tss <= 0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    return 1.0 - rss / tss

def _five_from_three(total, gdm, green):
    clover = gdm - green
    dead   = total - gdm
    return {
        'Dry_Green_g':  green,
        'Dry_Clover_g': clover,
        'Dry_Dead_g':   dead,
        'GDM_g':        gdm,
        'Dry_Total_g':  total,
    }

def csiro_weighted_r2_from_three_tensors(
    p_total, p_gdm, p_green,
    t_total, t_gdm, t_green,
    use_log1p=True
):
    """
    3ターゲットのTensor([N])から、5ターゲットに拡張して重み付きR²を返す。
    戻り: overall(float), per_target(dict)
    """
    to_np = lambda x: x.detach().cpu().numpy()
    # log1p学習なら実数空間に戻してから評価
    if use_log1p:
        p_total, p_gdm, p_green = [torch.expm1(x) for x in (p_total, p_gdm, p_green)]
        t_total, t_gdm, t_green = [torch.expm1(x) for x in (t_total, t_gdm, t_green)]

    y_true = _five_from_three(to_np(t_total), to_np(t_gdm), to_np(t_green))
    y_pred = _five_from_three(to_np(p_total), to_np(p_gdm), to_np(p_green))

    per = {k: _r2_1d(y_true[k], y_pred[k]) for k in y_true.keys()}
    wsum = sum(CSIRO_WEIGHTS.values())
    overall = float(np.nansum([CSIRO_WEIGHTS[k]/wsum * per[k] for k in per.keys()]))
    return overall, per


def csiro_weighted_r2_from_five_tensors(
    pred5: torch.Tensor,
    true5: torch.Tensor,
    columns=FIVE_TARGET_ORDER,
    use_log1p=True
):
    """
    5ターゲット（列順 columns）から重み付きR²を返す。
    戻り: overall(float), per_target(dict)
    """
    if use_log1p:
        pred5 = torch.expm1(pred5)
        true5 = torch.expm1(true5)

    to_np = lambda x: x.detach().cpu().numpy()
    p = to_np(pred5)
    t = to_np(true5)

    per = {}
    for j, name in enumerate(columns):
        per[name] = _r2_1d(t[:, j], p[:, j])
    wsum = sum(CSIRO_WEIGHTS.values())
    overall = float(np.nansum([CSIRO_WEIGHTS[k]/wsum * per[k] for k in columns]))
    return overall, per


def build_five_from_three_tensors(pred_tuple, target, use_log1p=True):
    """
    3出力のテンソルから、学習用の5出力テンソルを構築する。
    - 入力: pred_tuple = (pT, pGDM, pGR) 各[B]
            target: [B,3] (列順は Total, GDM, Green)
    - 変換は実数空間で行う（log1p学習時は expm1 で戻す）
    - 出力: (pred5[B,5], target5[B,5]) with order = FIVE_TARGET_ORDER
    """
    pT, pGDM, pGR = pred_tuple
    tT, tGDM, tGR = target[:,0], target[:,1], target[:,2]

    if use_log1p:
        PT = torch.expm1(pT); PG = torch.expm1(pGDM); PR = torch.expm1(pGR)
        TT = torch.expm1(tT); TG = torch.expm1(tGDM); TR = torch.expm1(tGR)
    else:
        PT, PG, PR = pT, pGDM, pGR
        TT, TG, TR = tT, tGDM, tGR

    # 5出力（Green, Dead, Clover, GDM, Total）
    pred_dead = PT - PG
    pred_clover = PG - PR
    tgt_dead = TT - TG
    tgt_clover = TG - TR

    pred_map = {
        'Dry_Green_g': PR,
        'Dry_Dead_g': pred_dead,
        'Dry_Clover_g': pred_clover,
        'GDM_g': PG,
        'Dry_Total_g': PT,
    }
    tgt_map = {
        'Dry_Green_g': TR,
        'Dry_Dead_g': tgt_dead,
        'Dry_Clover_g': tgt_clover,
        'GDM_g': TG,
        'Dry_Total_g': TT,
    }

    pred5 = torch.stack([pred_map[k] for k in FIVE_TARGET_ORDER], dim=1)
    tgt5 = torch.stack([tgt_map[k] for k in FIVE_TARGET_ORDER], dim=1)
    return pred5, tgt5

def train_one_fold(df, fold, lr=None, batch_size=None, wd=None, warmup_epochs=None, dropout=None, aug_strength=None):
    """
    Train one fold with optional parameter overrides.
    
    Args:
        df: DataFrame with fold column
        fold: Fold number to train
        lr: Learning rate override (default: CFG.LR)
        batch_size: Batch size override (default: CFG.BATCH_SIZE)
        wd: Weight decay override (default: CFG.WD)
        warmup_epochs: Warmup epochs override (default: CFG.WARMUP_EPOCHS)
        dropout: Dropout rate override (default: 0.3)
        aug_strength: Augmentation strength override (default: 1.0)
    """
    # Use overrides if provided, otherwise use CFG defaults
    train_lr = lr if lr is not None else CFG.LR
    train_batch_size = batch_size if batch_size is not None else CFG.BATCH_SIZE
    train_wd = wd if wd is not None else CFG.WD
    train_warmup_epochs = warmup_epochs if warmup_epochs is not None else CFG.WARMUP_EPOCHS
    train_dropout = dropout if dropout is not None else 0.3
    train_aug_strength = aug_strength if aug_strength is not None else 1.0
    
    print(f"\n===== FOLD {fold} / {CFG.N_FOLDS} =====")
    print(f"  LR={train_lr:.2e}, BATCH_SIZE={train_batch_size}, WD={train_wd:.4f}")
    print(f"  WARMUP_EPOCHS={train_warmup_epochs}, DROPOUT={train_dropout:.2f}, AUG_STRENGTH={train_aug_strength:.2f}")
    trn_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)

    train_tf = get_train_tf(CFG.IMG_SIZE, aug_strength=train_aug_strength)
    valid_tf = get_valid_tf(CFG.IMG_SIZE)

    trn_ds = TrainDataset(trn_df, CFG.TRAIN_IMAGE_DIR, train_tf, use_log1p=CFG.USE_LOG1P)
    val_ds = TrainDataset(val_df, CFG.TRAIN_IMAGE_DIR, valid_tf, use_log1p=CFG.USE_LOG1P)

    # Deterministic worker seeding
    def seed_worker(worker_id):
        s = torch.initial_seed() % 2**32
        np.random.seed(s)
        random.seed(s)
    g = torch.Generator()
    g.manual_seed(CFG.SEED)

    trn_dl = DataLoader(
        trn_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(CFG.NUM_WORKERS > 0),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=train_batch_size*2,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(CFG.NUM_WORKERS > 0),
    )

    model = BiomassModel(CFG.MODEL_NAME, pretrained=True, target_names=CFG.TARGETS, dual_stream=CFG.DUAL_STREAM, dropout=train_dropout).to(CFG.DEVICE)
    optimizer = AdamW(model.parameters(), lr=train_lr, weight_decay=train_wd)
    # True warmup + cosine, stepped per optimizer step
    steps_per_epoch = max(1, math.ceil(len(trn_dl) / CFG.GRAD_ACCUM))
    warmup_steps = max(1, train_warmup_epochs * steps_per_epoch)
    total_steps = max(1, CFG.EPOCHS * steps_per_epoch)
    cosine_steps = max(1, total_steps - warmup_steps)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=train_lr*1e-2)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.USE_AMP)
    ema = ModelEMA(model, decay=CFG.EMA_DECAY) if CFG.USE_EMA else None

    # 損失の重み（CFG.TARGETSの順序に基づく）
    weights = [CSIRO_WEIGHTS[k] for k in CFG.TARGETS]
    criterion = WeightedMSELoss(weights=weights)

    select_is_r2 = CFG.SELECT_BEST_BY.lower() == 'r2'
    best_metric = -float('inf') if select_is_r2 else float('inf')
    best_preds = None

    global_step = 0
    for epoch in range(1, CFG.EPOCHS+1):
        model.train()
        train_loss = 0.0

        for i, (l, r, y) in enumerate(tqdm(trn_dl, desc=f"Train ep{epoch}")):
            l = l.to(CFG.DEVICE, non_blocking=True); r = r.to(CFG.DEVICE, non_blocking=True); y = y.to(CFG.DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=CFG.USE_AMP):
                pred = model(l, r)  # [B,K]
                loss = criterion(pred, y) / CFG.GRAD_ACCUM
            scaler.scale(loss).backward()

            if (i+1) % CFG.GRAD_ACCUM == 0:
                if CFG.MAX_NORM is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema: ema.update(model)
                scheduler.step()
                global_step += 1

            train_loss += loss.item()

        # ---- validation (EMA優先) ----
        model.eval()
        eval_model = ema.ema if ema else model
        val_loss = 0.0
        y_pred_all, y_true_all = [], []

        with torch.no_grad():
            for l, r, y in tqdm(val_dl, desc="Valid"):
                l = l.to(CFG.DEVICE, non_blocking=True); r = r.to(CFG.DEVICE, non_blocking=True); y = y.to(CFG.DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=CFG.USE_AMP):
                    pred = eval_model(l, r)  # [B,K]
                    loss = criterion(pred, y)
                val_loss += loss.item()
                y_pred_all.append(pred.detach().cpu())  # [B,K]
                y_true_all.append(y.detach().cpu())

        y_pred_all = torch.cat(y_pred_all, dim=0)  # [N,K]
        y_true_all = torch.cat(y_true_all, dim=0)  # [N,K]

        # --- 既存: RMSE（参考） ---
        # RMSE（参考）: 常に Total/GDM/Green の3指標で計算
        idxT = CFG.TARGETS.index('Dry_Total_g')
        idxG = CFG.TARGETS.index('GDM_g')
        idxR = CFG.TARGETS.index('Dry_Green_g')
        rmse_mean, (rmse_T, rmse_G, rmse_R) = metric_rmse(
            (y_pred_all[:,idxT], y_pred_all[:,idxG], y_pred_all[:,idxR]),
            torch.stack([y_true_all[:,idxT], y_true_all[:,idxG], y_true_all[:,idxR]], dim=1),
            use_log1p=CFG.USE_LOG1P
        )

        if len(CFG.TARGETS) == 5:
            wr2, per_r2 = csiro_weighted_r2_from_five_tensors(
                pred5=y_pred_all,
                true5=y_true_all,
                columns=CFG.TARGETS,
                use_log1p=CFG.USE_LOG1P
            )
        else:
            wr2, per_r2 = csiro_weighted_r2_from_three_tensors(
                p_total=y_pred_all[:,0], p_gdm=y_pred_all[:,1], p_green=y_pred_all[:,2],
                t_total=y_true_all[:,0], t_gdm=y_true_all[:,1], t_green=y_true_all[:,2],
                use_log1p=CFG.USE_LOG1P
            )

        print(f"[Fold {fold}] Epoch {epoch}: "
              f"train_loss={train_loss/len(trn_dl):.4f}  "
              f"val_loss={val_loss/len(val_dl):.4f}  "
              f"RMSE_mean={rmse_mean:.4f} (T:{rmse_T:.4f} G:{rmse_G:.4f} R:{rmse_R:.4f})  "
              f"WeightedR2={wr2:.5f}  "
              f"R2(Total:{per_r2['Dry_Total_g']:.3f} GDM:{per_r2['GDM_g']:.3f} "
              f"Green:{per_r2['Dry_Green_g']:.3f} Dead:{per_r2['Dry_Dead_g']:.3f} "
              f"Clover:{per_r2['Dry_Clover_g']:.3f})"
        )

        current_metric = wr2 if select_is_r2 else rmse_mean.item()
        improved = (current_metric > best_metric) if select_is_r2 else (current_metric < best_metric)
        if epoch == 1:
            improved = True

        if improved:
            best_path = os.path.join(CFG.OUT_DIR, f'best_model_fold{fold}.pth')
            save_checkpoint(eval_model, best_path)
            best_metric = current_metric
            best_preds = y_pred_all.numpy()  # OOF保存用（ベスト）
            print(f"  -> Best updated ({CFG.SELECT_BEST_BY.upper()}). Save {best_path}")

    # Save OOF from best epoch
    oof_preds = best_preds if best_preds is not None else y_pred_all.numpy()
    if CFG.USE_LOG1P:
        oof_preds = np.expm1(oof_preds)
    oof_df = val_df[['image_path'] + CFG.TARGETS].copy()
    for i,t in enumerate(CFG.TARGETS):
        oof_df[f'pred_{t}'] = oof_preds[:, i]
    oof_df.to_csv(os.path.join(CFG.OUT_DIR, f'oof_fold{fold}.csv'), index=False)
    return best_metric
