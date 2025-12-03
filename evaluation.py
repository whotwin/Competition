import numpy as np
from utils import gwml_calculation

def SSres_calculation(weight_cls, labels, preds):
    res = (labels - preds) ** 2
    weighted_res = weight_cls * res
    weighted_sum = weighted_res.sum()
    return weighted_sum

def SStot_calculation(weight_cls, global_weighted_mean_labels, labels):
    res = (labels - global_weighted_mean_labels) ** 2
    weighted_res = weight_cls * res
    weighted_sum = weighted_res.sum()
    return weighted_sum

def r2_calculation(cfg, labels, preds, global_weighted_mean_label):
    weight_cls = cfg.weight_cls
    weight_cls = weight_cls.values()
    weight_cls = np.array(weight_cls).astype(np.float32)
    ssres = SSres_calculation(weight_cls, labels, preds)
    sstot = SStot_calculation(weight_cls, global_weighted_mean_label, labels)
    return 1 - np.sum(ssres / sstot)

def computing_metrics(cfg, preds, labels):
    assert labels.shape[-1] == 3
    a, b, c = labels[:, 0], labels[:, 1], labels[:, 2]
    full_labels = np.column_stack([labels[:, :3], a + b + c, a + c])
    a, b, c = preds[:, 0], preds[:, 1], preds[:, 2]
    full_preds = np.column_stack([preds[:, :3], a + b + c, a + c])
    weight_cls = cfg.weight_cls
    weight_cls = weight_cls.values()
    global_weighted_mean_label = gwml_calculation(weight_cls, full_labels)
    r2_score = r2_calculation(cfg, full_labels, full_preds, global_weighted_mean_label)
    return r2_score