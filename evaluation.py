import numpy as np

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