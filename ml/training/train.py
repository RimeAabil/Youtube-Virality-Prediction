import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, r2_score, classification_report
from features import FeatureExpert

def run_production_training():
    """Production training pipeline with enhanced virality scoring."""
    raw_path = "data/raw"
    models_path = "ml/models"
    os.makedirs(models_path, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Load Data
    try:
        print("Loading datasets...")
        df_v = pd.read_csv(f"{raw_path}/videos.csv")
        df_s = pd.read_csv(f"{raw_path}/statistics.csv")
        
        # Handle both 'snapshot_at' and 'captured_at' column names
        if 'snapshot_at' in df_s.columns and 'captured_at' not in df_s.columns:
            df_s = df_s.rename(columns={'snapshot_at': 'captured_at'})
            
        df_c = pd.read_csv(f"{raw_path}/channels.csv")
        print(f"✓ Loaded {len(df_v)} videos, {len(df_s)} statistics, {len(df_c)} channels")
        
    except FileNotFoundError as e:
        print(f"Error: Data files not found - {e}")
        print("Please run ingest_v2.py first to collect data.")
        return

    # Feature Engineering
    print("\nEngineering features...")
    fe = FeatureExpert()
    df = fe.prepare_dataset(df_v, df_s, df_c)
    
    print(f"✓ Dataset prepared with {len(df)} samples")
    print(f"✓ Viral videos: {df['is_viral_label'].sum()} ({df['is_viral_label'].mean()*100:.1f}%)")
    print(f"✓ Mean virality score: {df['virality_score'].mean():.2f}")
    
    # Feature Selection (Holistic & Precise)
    feature_cols = [
        # NLP Features
        'title_length', 'title_sentiment', 'emoji_count', 'contains_emojis',
        'is_uppercase', 'has_power_word', 'exclamation_count', 
        'contains_exclamation', 'question_count', 'all_caps_word_count', 
        'contains_number',
        
        # Temporal Features
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend',
        
        # Channel Features
        'subscriber_count_log',
        
        # Video Features
        'duration_seconds', 'legacy_engagement_score',
        
        # Enhanced Virality Components (for regression)
        'engagement_rate', 'velocity', 'reach_ratio', 'engagement_density'
    ]
    
    # Remove features that don't exist in current dataset
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"\nUsing {len(available_features)} features for training")
    
    X = df[available_features].fillna(0)
    
    # === 1. VIRALITY CLASSIFICATION (Binary: Viral or Not) ===
    print("\n" + "="*60)
    print("TRAINING VIRALITY CLASSIFIER (XGBoost)")
    print("="*60)
    
    y_class = df['is_viral_label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    clf = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate Classification
    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    
    print(f"\n📊 Classification Metrics:")
    print(f"  ROC-AUC:  {roc_auc_score(y_test, probs):.4f}")
    print(f"  F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Standard', 'Viral'])}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # === 2. VIRALITY SCORE REGRESSION (Continuous Score Prediction) ===
    print("\n" + "="*60)
    print("TRAINING VIRALITY SCORE REGRESSOR (LightGBM)")
    print("="*60)
    
    y_reg = df['virality_score']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
    reg = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        random_state=42
    )
    
    reg.fit(X_train_r, y_train_r)
    
    # Evaluate Regression
    preds_r = reg.predict(X_test_r)
    
    print(f"\n📊 Regression Metrics:")
    print(f"  MAE:  {mean_absolute_error(y_test_r, preds_r):.4f}")
    print(f"  R²:   {r2_score(y_test_r, preds_r):.4f}")
    print(f"  Mean Score (Actual):    {y_test_r.mean():.2f}")
    print(f"  Mean Score (Predicted): {preds_r.mean():.2f}")
    
    # Save Models and Metadata
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    joblib.dump(clf, f"{models_path}/virality_classifier_v2.joblib")
    joblib.dump(reg, f"{models_path}/timing_regressor_v2.joblib")
    joblib.dump(available_features, f"{models_path}/features_metadata.joblib")
    # Don't save FeatureExpert - will be recreated in predict.py to avoid import issues
    
    # Save processed data for analysis
    df.to_csv("data/processed/training_data.csv", index=False)
    feature_importance.to_csv("data/processed/feature_importance.csv", index=False)
    
    print(f"✓ Models saved to {models_path}/")
    print(f"✓ Processed data saved to data/processed/")
    print(f"✓ Feature importance saved to data/processed/feature_importance.csv")
    
    # Summary Statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total samples:        {len(df)}")
    print(f"Viral videos:         {df['is_viral_label'].sum()}")
    print(f"Standard videos:      {len(df) - df['is_viral_label'].sum()}")
    print(f"Features used:        {len(available_features)}")
    print(f"Classifier ROC-AUC:   {roc_auc_score(y_test, probs):.4f}")
    print(f"Regressor R²:         {r2_score(y_test_r, preds_r):.4f}")
    print("="*60)

if __name__ == "__main__":
    run_production_training()