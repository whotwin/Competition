class CFG:
    train_csv_path = './csiro-biomass/train.csv'
    data_root = './csiro-biomass/'
    weight_cls = {'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1, 'GDM_g': 0.2, 'Dry_Total_g': 0.5}
    target_order = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']

    batch_size = 8
    num_workers = 4