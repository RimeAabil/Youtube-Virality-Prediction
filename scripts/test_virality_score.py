
import pandas as pd
import numpy as np
import importlib.util
import sys
import os

# Load the module dynamically to handle the space and parentheses in filename
file_path = r'c:\Users\MOHSEN\Documents\ELASRI\youtube_eda (1).py'
module_name = 'youtube_eda_mod'

try:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    print(f"Successfully loaded {file_path}")
except Exception as e:
    print(f"Error loading module: {e}")
    sys.exit(1)

# Create dummy data for testing
print("\nGenerating dummy data...")
data = {
    'view_count': [1500, 10000, 50, 1000000, 2500],
    'like_count': [100, 500, 2, 50000, 150],
    'comment_count': [20, 100, 0, 1000, 10],
    'dislike_count': [5, 10, 0, 500, 2], # Testing with dislike_count present
    'duration_seconds': [120, 600, 30, 300, 180],
    'published_at': [
        pd.Timestamp.now() - pd.Timedelta(days=2),
        pd.Timestamp.now() - pd.Timedelta(days=30),
        pd.Timestamp.now() - pd.Timedelta(days=1),
        pd.Timestamp.now() - pd.Timedelta(days=365),
        pd.Timestamp.now() - pd.Timedelta(days=7)
    ],
    'subscriber_count': [500, 5000, 10, 100000, 1000]
}

df = pd.DataFrame(data)
print("Input DataFrame:")
print(df[['view_count', 'like_count', 'published_at']])

# Run the function
print("\nExecuting compute_virality_score_v2...")
try:
    df_result = module.compute_virality_score_v2(df)
    
    print("\nExecution Successful!")
    print("\nResulting Virality Scores (Top 5):")
    print(df_result[['view_count', 'engagement_rate', 'velocity', 'virality_score', 'is_viral_label']])
    
    # Check if viral label works (should be 1 for the top one)
    print("\nViral Label Distribution:")
    print(df_result['is_viral_label'].value_counts())
    
except Exception as e:
    print(f"Error executing function: {e}")
    # Print traceback for easier debugging if needed
    import traceback
    traceback.print_exc()
