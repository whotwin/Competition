import timm
import torch
import numpy as np
import torch.nn as nn
from .config import CFG

class BiomassModel(nn.Module):
    def __init__(self, model_name='convnext_tiny', pretrained=True, target_names=None, dual_stream=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg', in_chans=3)
        self.target_names = target_names if target_names is not None else ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
        self.num_outputs = len(self.target_names)
        self.dual_stream = dual_stream
        self.dropout = dropout
        nf = self.backbone.num_features
        self.n_combined_features = nf * 2 if self.dual_stream else nf

        for target_name in self.target_names:
            head = nn.Sequential(
                nn.Linear(self.n_combined_features, self.n_combined_features // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.n_combined_features // 2, 1)
            )
            setattr(self, f'head_{target_name.lower().replace("_", "")}', head)

    def forward(self, l, r=None):
        fl = self.backbone(l)
        if self.dual_stream:
            fr = self.backbone(r)
            x = torch.cat([fl, fr], dim=1)
        else:
            x = fl

        outputs = []
        for target_name in self.target_names:
            head = getattr(self, f'head_{target_name.lower().replace("_", "")}')
            out = head(x).squeeze(1)  # [B,1] -> [B]
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)  # [B, num_outputs]
    
class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss for 3 targets (Total, GDM, Green).
    """
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, predictions, targets):
        self.weights = self.weights.to(predictions.device)
        # predictions/targets: [B, 3]
        mse_per_target = (predictions - targets) ** 2
        weighted_mse = mse_per_target * self.weights.unsqueeze(0)
        return weighted_mse.mean()

class ConstraintLoss(nn.Module):

    def __init__(self, l1_w=1.0, cons_w=0.1, nonneg_w=0.05, use_log1p=True):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_w = l1_w
        self.cons_w = cons_w
        self.nonneg_w = nonneg_w
        self.use_log1p = use_log1p

    def forward(self, pred, target):
        # pred/target: tuple of three tensors (B,)
        pT, pGDM, pGR = pred
        tT, tGDM, tGR = target[:,0], target[:,1], target[:,2]

        # 主損失：ログ空間ならそのままL1
        loss_main = self.l1(pT, tT) + self.l1(pGDM, tGDM) + self.l1(pGR, tGR)
        loss_main = self.l1_w * loss_main / 3.0

        # 制約は実数空間で評価
        if self.use_log1p:
            PT = torch.expm1(pT)
            PG = torch.expm1(pGDM)
            PR = torch.expm1(pGR)
        else:
            PT, PG, PR = pT, pGDM, pGR

        zero = torch.zeros_like(PT)
        # monotonic violation
        v1 = torch.relu(PG - PT)   # want PT >= PG
        v2 = torch.relu(PR - PG)   # want PG >= PR
        loss_cons = (v1 + v2).mean() * self.cons_w

        # non-negative violation
        n1 = torch.relu(-PT); n2 = torch.relu(-PG); n3 = torch.relu(-PR)
        loss_nonneg = (n1 + n2 + n3).mean() * self.nonneg_w

        return loss_main + loss_cons + loss_nonneg

def rmse_torch(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

def metric_rmse(pred_tuple, target, use_log1p=True):
    pT, pGDM, pGR = pred_tuple
    tT, tGDM, tGR = target[:,0], target[:,1], target[:,2]
    if use_log1p:
        pT, pGDM, pGR = [torch.expm1(x) for x in (pT, pGDM, pGR)]
        tT, tGDM, tGR = [torch.expm1(x) for x in (tT, tGDM, tGR)]
    rmse_T = rmse_torch(pT, tT)
    rmse_G = rmse_torch(pGDM, tGDM)
    rmse_R = rmse_torch(pGR, tGR)
    return (rmse_T + rmse_G + rmse_R) / 3.0, (rmse_T, rmse_G, rmse_R)

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = BiomassModel(CFG.MODEL_NAME, pretrained=False, target_names=model.target_names, 
                                dual_stream=model.dual_stream, dropout=model.dropout).to(CFG.DEVICE)
        self.ema.load_state_dict(model.state_dict())
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            v.copy_(v * d + (1. - d) * msd[k])