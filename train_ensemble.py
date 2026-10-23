import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from scipy.optimize import minimize
import time
import os

def main():
    print("=" * 70)
    print(">>> PLANET HUNTERS: FULL TRIPLE GRADIENT BOOSTING ENSEMBLE TRAINING <<<")
    print("=" * 70)

    t_start = time.time()
    
    # 1. Load Data
    train_path = "train_features_v2.parquet"
    test_path = "test_features_v2.parquet"
    
    assert os.path.exists(train_path), f"Missing {train_path}"
    assert os.path.exists(test_path), f"Missing {test_path}"
    
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    test_stars = pd.read_csv("test_stars.csv")
    
    feature_cols = [c for c in df_train.columns if c not in ['id', 'label']]
    print(f"Loaded {len(df_train)} train stars, {len(df_test)} test stars.")
    print(f"Number of engineered features: {len(feature_cols)}")
    
    labels = sorted(df_train['label'].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    
    print("\nClass mapping:")
    for l, i in label2id.items():
        count = (df_train['label'] == l).sum()
        pct = count / len(df_train) * 100
        print(f"  {i}: {l:<20s} ({count:5d} stars, {pct:5.2f}%)")
        
    X = df_train[feature_cols].copy()
    y = df_train['label'].map(label2id).values
    X_test = df_test[feature_cols].copy()
    
    # 2. Stratified 5-Fold Setup
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    oof_lgb = np.zeros((len(df_train), len(labels)))
    oof_xgb = np.zeros((len(df_train), len(labels)))
    oof_cb  = np.zeros((len(df_train), len(labels)))
    oof_ens = np.zeros((len(df_train), len(labels)))
    
    test_preds_lgb = np.zeros((len(df_test), len(labels)))
    test_preds_xgb = np.zeros((len(df_test), len(labels)))
    test_preds_cb  = np.zeros((len(df_test), len(labels)))
    test_preds_ens = np.zeros((len(df_test), len(labels)))
    
    w_lgb, w_xgb, w_cb = 0.40, 0.35, 0.25
    
    print("\n" + "=" * 70)
    print(f"Starting {N_FOLDS}-Fold Cross-Validation with Ensemble (LGB: {w_lgb}, XGB: {w_xgb}, CB: {w_cb})")
    print("=" * 70)
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        f_start = time.time()
        print(f"\n--- FOLD {fold + 1} / {N_FOLDS} ---")
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        
        # A. LightGBM
        print("  Training LightGBM...")
        model_lgb = lgb.LGBMClassifier(
            n_estimators=550,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )
        model_lgb.fit(X_tr, y_tr)
        p_va_lgb = model_lgb.predict_proba(X_va)
        p_te_lgb = model_lgb.predict_proba(X_test)
        oof_lgb[va_idx] = p_va_lgb
        test_preds_lgb += p_te_lgb / N_FOLDS
        f1_lgb = f1_score(y_va, np.argmax(p_va_lgb, 1), average='macro')
        print(f"  -> LightGBM Fold {fold + 1} Macro-F1: {f1_lgb:.4f}")
        
        # B. XGBoost
        print("  Training XGBoost...")
        model_xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.035,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42 + fold,
            n_jobs=-1
        )
        model_xgb.fit(X_tr, y_tr)
        p_va_xgb = model_xgb.predict_proba(X_va)
        p_te_xgb = model_xgb.predict_proba(X_test)
        oof_xgb[va_idx] = p_va_xgb
        test_preds_xgb += p_te_xgb / N_FOLDS
        f1_xgb = f1_score(y_va, np.argmax(p_va_xgb, 1), average='macro')
        print(f"  -> XGBoost Fold {fold + 1} Macro-F1:  {f1_xgb:.4f}")
        
        # C. CatBoost
        print("  Training CatBoost...")
        model_cb = CatBoostClassifier(
            iterations=650,
            learning_rate=0.045,
            depth=6,
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1
        )
        model_cb.fit(X_tr, y_tr)
        p_va_cb = model_cb.predict_proba(X_va)
        p_te_cb = model_cb.predict_proba(X_test)
        oof_cb[va_idx] = p_va_cb
        test_preds_cb += p_te_cb / N_FOLDS
        f1_cb = f1_score(y_va, np.argmax(p_va_cb, 1), average='macro')
        print(f"  -> CatBoost Fold {fold + 1} Macro-F1: {f1_cb:.4f}")
        
        # Fold Blend
        p_va_ens = w_lgb * p_va_lgb + w_xgb * p_va_xgb + w_cb * p_va_cb
        p_te_ens = w_lgb * p_te_lgb + w_xgb * p_te_xgb + w_cb * p_te_cb
        oof_ens[va_idx] = p_va_ens
        test_preds_ens += p_te_ens / N_FOLDS
        f1_ens = f1_score(y_va, np.argmax(p_va_ens, 1), average='macro')
        print(f"  [+] Fold {fold + 1} ENSEMBLE Macro-F1: {f1_ens:.4f}  (Time: {time.time() - f_start:.1f}s)")

    # 3. Overall CV Evaluation
    print("\n" + "=" * 70)
    print("OVERALL 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)
    
    total_lgb = f1_score(y, np.argmax(oof_lgb, 1), average='macro')
    total_xgb = f1_score(y, np.argmax(oof_xgb, 1), average='macro')
    total_cb  = f1_score(y, np.argmax(oof_cb, 1), average='macro')
    total_ens = f1_score(y, np.argmax(oof_ens, 1), average='macro')
    
    print(f"Overall LightGBM Macro-F1: {total_lgb:.4f}")
    print(f"Overall XGBoost  Macro-F1: {total_xgb:.4f}")
    print(f"Overall CatBoost Macro-F1: {total_cb:.4f}")
    print(f"Overall ENSEMBLE Raw Macro-F1: {total_ens:.4f}")
    
    pred_names = [id2label[i] for i in np.argmax(oof_ens, 1)]
    true_names = [id2label[i] for i in y]
    print("\nRaw Ensemble Classification Report:")
    print(classification_report(true_names, pred_names, digits=4))
    
    # 4. Threshold & Class Multiplier Optimization
    print("\n" + "=" * 70)
    print("OPTIMIZING CLASS MULTIPLIERS FOR MACRO-F1")
    print("=" * 70)
    
    def loss_func(weights):
        if np.any(weights <= 0.05):
            return 999.0
        scaled = oof_ens * weights
        pred = np.argmax(scaled, axis=1)
        return -f1_score(y, pred, average='macro')
    
    init_weights = np.ones(len(labels))
    opt_res = minimize(loss_func, init_weights, method='Powell', options={'maxiter': 1000})
    best_weights = opt_res.x
    best_weights = best_weights / np.mean(best_weights)
    
    oof_opt_preds = np.argmax(oof_ens * best_weights, axis=1)
    best_macro_f1 = f1_score(y, oof_opt_preds, average='macro')
    print(f"OPTIMIZED ENSEMBLE MACRO-F1: {best_macro_f1:.4f} (gain: +{best_macro_f1 - total_ens:.4f})")
    
    print("\nOptimal class weights:")
    for l, w in zip(labels, best_weights):
        print(f"  {l:<20s}: {w:.3f}")
        
    print("\nOptimized Ensemble Classification Report:")
    opt_pred_names = [id2label[i] for i in oof_opt_preds]
    print(classification_report(true_names, opt_pred_names, digits=4))
    
    # 5. Generate Submission Files
    print("\n" + "=" * 70)
    print("GENERATING TEST SUBMISSION FILES")
    print("=" * 70)
    
    # A. Optimized Submission (Primary)
    test_opt_probs = test_preds_ens * best_weights
    test_opt_pred_ids = np.argmax(test_opt_probs, axis=1)
    test_opt_labels = [id2label[i] for i in test_opt_pred_ids]
    
    sub_df = pd.DataFrame({
        'id': test_stars['id'],
        'label': test_opt_labels
    })
    sub_df.to_csv("submission.csv", index=False)
    print("Saved primary submission to: submission.csv")
    
    # B. Raw Ensemble Submission (Backup)
    test_raw_pred_ids = np.argmax(test_preds_ens, axis=1)
    test_raw_labels = [id2label[i] for i in test_raw_pred_ids]
    sub_raw_df = pd.DataFrame({
        'id': test_stars['id'],
        'label': test_raw_labels
    })
    sub_raw_df.to_csv("submission_raw_ensemble.csv", index=False)
    print("Saved backup raw ensemble to: submission_raw_ensemble.csv")
    
    # 6. Strict Quality & Compliance Checks
    print("\n" + "=" * 70)
    print("RUNNING COMPLIANCE VERIFICATION ON submission.csv")
    print("=" * 70)
    
    sub_check = pd.read_csv("submission.csv")
    
    # Check 1: Shape
    assert sub_check.shape == (7000, 2), f"Incorrect shape: {sub_check.shape}, expected (7000, 2)"
    print("Check 1 passed: Exactly 7,000 rows and 2 columns.")
    
    # Check 2: Columns
    assert list(sub_check.columns) == ['id', 'label'], f"Incorrect columns: {list(sub_check.columns)}"
    print("Check 2 passed: Exact column headers ['id', 'label'].")
    
    # Check 3: IDs match test_stars.csv exactly
    assert (sub_check['id'] == test_stars['id']).all(), "Test IDs do not match test_stars.csv!"
    print("Check 3 passed: All 7,000 test IDs match test_stars.csv in exact order.")
    
    # Check 4: No nulls / NaNs
    assert sub_check['id'].isna().sum() == 0, "Found NaNs in 'id' column!"
    assert sub_check['label'].isna().sum() == 0, "Found NaNs in 'label' column!"
    print("Check 4 passed: Zero NaNs or missing values.")
    
    # Check 5: All labels are valid
    valid_labels_set = set(labels)
    assert set(sub_check['label'].unique()).issubset(valid_labels_set), "Found invalid labels!"
    print("Check 5 passed: All predicted labels are valid hackathon class names.")
    
    # Check 6: Label distribution
    print("\nPredicted Class Distribution in submission.csv:")
    dist = sub_check['label'].value_counts()
    for l in labels:
        cnt = dist.get(l, 0)
        pct = cnt / len(sub_check) * 100
        print(f"  {l:<20s}: {cnt:5d} ({pct:5.2f}%)")
        
    print("\n" + "=" * 70)
    print(f"SUCCESS: Pipeline completed in {time.time() - t_start:.1f}s!")
    print("=" * 70)

if __name__ == '__main__':
    main()
