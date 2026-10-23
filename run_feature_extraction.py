import os
import time
import pandas as pd
import numpy as np
from features import extract_features_dataset

def main():
    print("=== STARTING FEATURE EXTRACTION V2 (WITH CCD COTRENDING & DEEP ASTRONOMICAL FEATURES) ===")
    
    train_feat_path = "train_features_v2.parquet"
    test_feat_path = "test_features_v2.parquet"
    
    if os.path.exists(train_feat_path):
        print(f"{train_feat_path} already exists! Skipping extraction.")
    else:
        print("Extracting train features v2...")
        t0 = time.time()
        df_train = extract_features_dataset(
            'train_flux.npy', 'train_centroid.npy', 'train_stars.csv', 'train_cadences.csv', n_jobs=14
        )
        train_stars = pd.read_csv('train_stars.csv')
        df_train['id'] = train_stars['id']
        df_train['label'] = train_stars['label']
        df_train.to_parquet(train_feat_path, index=False)
        print(f"Train features v2 saved to {train_feat_path} in {time.time() - t0:.1f}s. Shape: {df_train.shape}")
        
    if os.path.exists(test_feat_path):
        print(f"{test_feat_path} already exists! Skipping extraction.")
    else:
        print("Extracting test features v2...")
        t0 = time.time()
        df_test = extract_features_dataset(
            'test_flux.npy', 'test_centroid.npy', 'test_stars.csv', 'test_cadences.csv', n_jobs=14
        )
        test_stars = pd.read_csv('test_stars.csv')
        df_test['id'] = test_stars['id']
        df_test.to_parquet(test_feat_path, index=False)
        print(f"Test features v2 saved to {test_feat_path} in {time.time() - t0:.1f}s. Shape: {df_test.shape}")

    print("=== ALL FEATURE EXTRACTION V2 COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()
