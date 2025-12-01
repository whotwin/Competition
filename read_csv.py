import time
import numpy as np
import pandas as pd

def read_csv(data_path):
    df = pd.read_csv(data_path)
    print(df.columns.values)
    df_np = df.values
    length, dim = df_np.shape
    print(f"Total {length} records, each record contains {dim} features.\n")
    print("-"*50)
    return df_np, df

if __name__ == '__main__':
    data_root = './csiro-biomass/train.csv'
    df_np, df = read_csv(data_root)
    data = df['image_path'][1]
    time.sleep(1)