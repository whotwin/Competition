import os
import torch

class CFG:
    # paths
    SELECT_BEST_BY = 'r2' 
    BASE_PATH = '/kaggle/input/csiro-biomass'
    TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
    TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train')
    OUT_DIR = '.'           # 重み・OOFの保存先
    os.makedirs(OUT_DIR, exist_ok=True)

    # model
    MODEL_NAME = 'convnext_tiny'
    IMG_SIZE = 512
    IN_CHANS = 3
    DUAL_STREAM = True  # True: 左右二流, False: 単一ストリーム（全画像）

    # folds
    N_FOLDS = 5
    SEED = 42

    # train setup
    EPOCHS = 80
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    LR = 3e-4
    WD = 0.05
    WARMUP_EPOCHS = 1
    GRAD_ACCUM = 1
    MAX_NORM = 1.0
    USE_AMP = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loss / targets
    TARGETS = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    TRAIN_FIVE_OUTPUT_LOSS = False   # 5出力（Green, Dead, Clover, GDM, Total）で学習するか
    USE_LOG1P = False             # log1p変換で安定化

    # ema
    USE_EMA = False
    EMA_DECAY = 0.999

    # deterministic
    DETERMINISTIC = True

    # inference
    INFERENCE_MODE =  False  # False=学習, True=推論
    INFERENCE_MODEL_DIR = "/kaggle/input/convnext-tiny"  # NoneならOUT_DIRを使用、指定時はそのディレクトリからモデルを読み込み
    INFERENCE_BATCH_SIZE = 32
    USE_TTA = False  # Test-time augmentation
    TEST_CSV = os.path.join(BASE_PATH, 'test.csv')
    TEST_IMAGE_DIR = os.path.join(BASE_PATH, 'test')
    SUBMISSION_OUTPUT = os.path.join(OUT_DIR, 'submission.csv')
    INFERENCE_FOLDS = None  # Noneなら全fold自動検出、リスト指定時はそのfoldのみ使用