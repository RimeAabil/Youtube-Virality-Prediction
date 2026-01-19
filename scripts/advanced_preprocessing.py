# -*- coding: utf-8 -*-
"""
Advanced Preprocessing Module for YouTube Video Virality Prediction
This module contains all feature engineering and preprocessing functions
"""

import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPE FIXING
# =============================================================================

def fix_data_types(df):
    """
    Ensure all numeric columns are properly typed
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with corrected data types
    """
    logger.info("Fixing data types...")
    
    df = df.copy()
    
    # List of columns that should be numeric
    numeric_columns = [
        'view_count', 'like_count', 'comment_count', 
        'favorite_count', 'duration', 'subscriber_count',
        'video_count', 'view_count_channel'
    ]
    
    # Convert to numeric, handling errors gracefully
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info(f"Converted {col} to numeric type")
    
    # Ensure datetime columns are properly typed
    datetime_columns = ['published_at']
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Remove timezone if present
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
            logger.info(f"Converted {col} to datetime type")
    
    # Fill NaN values in numeric columns
    for col in numeric_columns:
        if col in df.columns and df[col].isna().any():
            if 'count' in col.lower():
                fill_value = df[col].median() if df[col].notna().any() else 0
            else:
                fill_value = 0
            df[col] = df[col].fillna(fill_value)
            logger.info(f"Filled NaN values in {col} with {fill_value}")
    
    logger.info("Data types fixed successfully")
    return df


# =============================================================================
# VIRALITY SCORE CALCULATION
# =============================================================================

def calculate_virality_score(df):
    """
    Calculate virality score with proper error handling
    
    Args:
        df: DataFrame with engagement metrics
        
    Returns:
        DataFrame with virality_score and is_viral columns
    """
    logger.info("Calculating virality score...")
    
    df = df.copy()
    
    # Ensure numeric types
    for col in ['view_count', 'like_count', 'comment_count']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    # Calculate engagement rate with safe division
    df['engagement_rate'] = 0.0
    if all(col in df.columns for col in ['view_count', 'like_count', 'comment_count']):
        view_count_safe = df['view_count'].replace(0, 1)
        df['engagement_rate'] = (
            (df['like_count'] + df['comment_count']) / view_count_safe
        ) * 100
    
    # Calculate virality score (normalized)
    if 'view_count' in df.columns:
        df['virality_score'] = np.log1p(df['view_count'])
        
        # Normalize to 0-100 scale
        if df['virality_score'].max() > 0:
            df['virality_score'] = (
                (df['virality_score'] - df['virality_score'].min()) / 
                (df['virality_score'].max() - df['virality_score'].min())
            ) * 100
    else:
        df['virality_score'] = 0.0
    
    # Define viral threshold (top 25% for dataset-relative virality)
    if 'view_count' in df.columns:
        # Use dataset-relative threshold (75th percentile)
        # This ensures we always have some viral videos for analysis
        threshold = df['view_count'].quantile(0.75)
        df['is_viral'] = (df['view_count'] >= threshold).astype(int)
        logger.info(f"Viral threshold set to: {threshold:,.0f} views")
    else:
        df['is_viral'] = 0
    
    logger.info(f"Calculated virality score. Viral videos: {df['is_viral'].sum()}")
    
    return df


# =============================================================================
# TITLE FEATURE EXTRACTION
# =============================================================================

def extract_title_features(df):
    """
    Extract features from video titles
    
    Args:
        df: DataFrame with 'title' column
        
    Returns:
        DataFrame with title features added
    """
    logger.info("Extracting title features...")
    
    df = df.copy()
    
    if 'title' in df.columns:
        # Basic length features
        df['title_length'] = df['title'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        
        # Punctuation features
        df['has_exclamation'] = df['title'].str.contains('!', regex=False).astype(int)
        df['has_question'] = df['title'].str.contains('\?', regex=True).astype(int)
        
        # Emoji detection (simple approach)
        emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
        df['emoji_count'] = df['title'].apply(lambda x: len(emoji_pattern.findall(str(x))))
        df['has_emoji'] = (df['emoji_count'] > 0).astype(int)
        
        # Capitalization features
        df['caps_word_count'] = df['title'].apply(
            lambda x: sum(1 for word in str(x).split() if word.isupper() and len(word) > 1)
        )
        
        # Number detection
        df['has_numbers'] = df['title'].str.contains(r'\d', regex=True).astype(int)
        
        # Special characters count
        df['special_char_count'] = df['title'].apply(
            lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', str(x)))
        )
        
        # Clickbait indicators (simple heuristic)
        clickbait_words = ['amazing', 'shocking', 'unbelievable', 'secret', 'revealed','must', 'never', 'always', 'everyone', 'nobody']
        df['clickbait_score'] = df['title'].str.lower().apply(
            lambda x: sum(word in str(x) for word in clickbait_words)
        )
        
        logger.info(f"Extracted {10} title features")
    else:
        logger.warning("'title' column not found. Skipping title features.")
    
    return df


# =============================================================================
# TEMPORAL FEATURE EXTRACTION
# =============================================================================

def extract_temporal_features(df):
    """
    Extract temporal features from published_at timestamp
    
    Args:
        df: DataFrame with 'published_at' column
        
    Returns:
        DataFrame with temporal features added
    """
    logger.info("Extracting temporal features...")
    
    df = df.copy()
    
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'])
        
        # Remove timezone information
        if df['published_at'].dt.tz is not None:
            df['published_at'] = df['published_at'].dt.tz_localize(None)
        
        current_date = datetime.now()
        
        # Extract basic temporal features
        df['publish_hour'] = df['published_at'].dt.hour
        df['publish_day'] = df['published_at'].dt.day
        df['publish_month'] = df['published_at'].dt.month
        df['publish_year'] = df['published_at'].dt.year
        df['publish_dayofweek'] = df['published_at'].dt.dayofweek
        
        # Binary features
        df['is_weekend'] = (df['publish_dayofweek'] >= 5).astype(int)
        df['is_peak_hour'] = df['publish_hour'].apply(
            lambda x: 1 if x in [17, 18, 19, 20, 21] else 0
        )
        
        # Days since publication
        df['days_since_publication'] = (current_date - df['published_at']).dt.days
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['publish_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['publish_hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['publish_dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['publish_dayofweek'] / 7)
        
        logger.info(f"Extracted 13 temporal features")
    else:
        logger.warning("'published_at' column not found. Skipping temporal features.")
    
    return df


# =============================================================================
# CHANNEL FEATURE EXTRACTION
# =============================================================================

def extract_channel_features(df, channel_df=None):
    """
    Extract channel-level features
    
    Args:
        df: Video DataFrame
        channel_df: Channel data DataFrame (optional)
        
    Returns:
        DataFrame with channel features added
    """
    logger.info("Extracting channel features...")
    
    df = df.copy()
    
    if channel_df is not None:
        # Merge channel data
        df = df.merge(channel_df, on='channel_id', how='left')
        logger.info("Channel data merged successfully")
    else:
        logger.warning("Channel data not provided. Skipping channel features.")
    
    return df


# =============================================================================
# VIDEO METADATA FEATURE EXTRACTION
# =============================================================================

def extract_video_metadata_features(df):
    """
    Extract features from video metadata
    
    Args:
        df: DataFrame with video metadata
        
    Returns:
        DataFrame with metadata features added
    """
    logger.info("Extracting video metadata features...")
    
    df = df.copy()
    
    # Duration features (if available)
    if 'duration' in df.columns:
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
        df['duration'] = df['duration'].fillna(0)
        
        # Duration categories
        df['is_short'] = (df['duration'] < 60).astype(int)  # Less than 1 min
        df['is_medium'] = ((df['duration'] >= 60) & (df['duration'] < 600)).astype(int)  # 1-10 min
        df['is_long'] = (df['duration'] >= 600).astype(int)  # More than 10 min
        
        logger.info("Extracted duration features")
    
    # Category features (if available)
    if 'category_id' in df.columns:
        # Keep as is for now, will be encoded later
        logger.info("Category ID found")
    
    return df


# =============================================================================
# INTERACTION FEATURE CREATION
# =============================================================================

def create_interaction_features(df):
    """
    Create interaction features with safe division
    
    Args:
        df: DataFrame with basic features
        
    Returns:
        DataFrame with interaction features added
    """
    logger.info("Creating interaction features...")
    
    df = df.copy()
    
    # Ensure numeric types
    for col in ['view_count', 'like_count', 'comment_count']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    # Ratio features with safe division
    if 'view_count' in df.columns and 'like_count' in df.columns:
        like_safe = df['like_count'].replace(0, 1)
        df['view_to_like_ratio'] = df['view_count'] / like_safe
    
    if 'view_count' in df.columns and 'comment_count' in df.columns:
        comment_safe = df['comment_count'].replace(0, 1)
        df['view_to_comment_ratio'] = df['view_count'] / comment_safe
    
    if 'like_count' in df.columns and 'comment_count' in df.columns:
        comment_safe = df['comment_count'].replace(0, 1)
        df['like_to_comment_ratio'] = df['like_count'] / comment_safe
    
    # Engagement score
    if all(col in df.columns for col in ['view_count', 'like_count', 'comment_count']):
        view_safe = df['view_count'].replace(0, 1)
        df['engagement_score'] = (
            (df['like_count'] * 2 + df['comment_count'] * 3) / view_safe
        ) * 100
    
    # Title length interactions
    if 'title_length' in df.columns and 'view_count' in df.columns:
        df['title_length_x_views'] = df['title_length'] * np.log1p(df['view_count'])
    
    # Temporal interactions
    if 'is_weekend' in df.columns and 'engagement_rate' in df.columns:
        df['weekend_engagement'] = df['is_weekend'] * df['engagement_rate']
    
    if 'is_peak_hour' in df.columns and 'engagement_rate' in df.columns:
        df['peak_hour_engagement'] = df['is_peak_hour'] * df['engagement_rate']
    
    logger.info("Created interaction features")
    
    return df


# =============================================================================
# MISSING VALUE HANDLING
# =============================================================================

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Args:
        df: DataFrame with potential missing values
        
    Returns:
        DataFrame with missing values handled
    """
    logger.info("Handling missing values...")
    
    df = df.copy()
    
    # Numeric columns - fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Categorical/Object columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            # Get fill value
            if df[col].mode().empty:
                fill_value = 'unknown'
            else:
                fill_value = df[col].mode()[0]
            
            # Handle categorical types properly
            if pd.api.types.is_categorical_dtype(df[col]):
                if fill_value not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories([fill_value])
                df[col] = df[col].fillna(fill_value)
            else:
                df[col] = df[col].fillna(fill_value)
    
    logger.info("Missing values handled successfully")
    return df


# =============================================================================
# OUTLIER REMOVAL
# =============================================================================

def remove_outliers(df, columns=None, threshold=3):
    """
    Remove outliers using IQR method
    
    Args:
        df: Input DataFrame
        columns: List of columns to check for outliers (None = all numeric)
        threshold: IQR multiplier for outlier detection (default=3, use 5 for less aggressive)
        
    Returns:
        DataFrame with outliers removed
    """
    logger.info("Removing outliers...")
    
    df = df.copy()
    original_len = len(df)
    
    if columns is None:
        # Only check specific columns that are prone to outliers
        columns = ['view_count', 'like_count', 'comment_count', 'duration_seconds']
        columns = [c for c in columns if c in df.columns]
    
    # Use a more lenient threshold for small datasets
    if original_len < 100:
        threshold = 5  # More lenient for small datasets
        logger.info(f"Using lenient threshold ({threshold}) for small dataset")
    
    for col in columns:
        if col in df.columns and col not in ['is_viral', 'is_weekend', 'is_peak_hour']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # Skip if no variance
                continue
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            before = len(df)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            after = len(df)
            
            if before != after:
                logger.info(f"  {col}: removed {before - after} outliers")
    
    removed = original_len - len(df)
    logger.info(f"Total removed: {removed} outliers ({removed/original_len*100:.2f}%)")
    
    return df


# =============================================================================
# FEATURE SCALING
# =============================================================================

def scale_features(df, exclude_cols=None):
    """
    Scale numeric features using StandardScaler
    
    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude from scaling
        
    Returns:
        DataFrame with scaled features
    """
    logger.info("Scaling features...")
    
    df = df.copy()
    
    if exclude_cols is None:
        exclude_cols = ['is_viral', 'video_id', 'title', 'published_at']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    
    if cols_to_scale:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        logger.info(f"Scaled {len(cols_to_scale)} features")
    
    return df


# =============================================================================
# CATEGORICAL ENCODING
# =============================================================================

def encode_categorical_features(df):
    """
    Encode categorical features
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with encoded categorical features
    """
    logger.info("Encoding categorical features...")
    
    df = df.copy()
    
    # Get categorical columns (excluding target and ID columns)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    exclude_cols = ['video_id', 'title', 'published_at', 'channel_id', 'description']
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    for col in categorical_cols:
        if df[col].nunique() < 50:  # Only encode if not too many unique values
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            logger.info(f"Encoded {col}")
    
    logger.info("Categorical encoding completed")
    return df


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

def preprocess_pipeline(video_df, channel_df=None, remove_outliers_flag=True, scale_features_flag=False):
    """
    Complete preprocessing pipeline
    
    Args:
        video_df: Video data DataFrame
        channel_df: Channel data DataFrame (optional)
        remove_outliers_flag: Whether to remove outliers (boolean)
        scale_features_flag: Whether to scale features (boolean)
        
    Returns:
        Fully preprocessed DataFrame
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Store flags to avoid name conflicts with functions
    should_remove_outliers = remove_outliers_flag
    should_scale_features = scale_features_flag
    
    # Step 0: Fix data types FIRST
    df = fix_data_types(video_df)
    
    # Step 1: Calculate virality score (before feature extraction)
    df = calculate_virality_score(df)
    
    # Step 2: Extract features
    df = extract_title_features(df)
    df = extract_temporal_features(df)
    df = extract_channel_features(df, channel_df)
    df = extract_video_metadata_features(df)
    df = create_interaction_features(df)
    
    # Step 3: Handle missing values
    df = handle_missing_values(df)
    
    # Step 4: Remove outliers if requested
    if should_remove_outliers:
        df = remove_outliers(df)
    
    # Step 5: Encode categorical features
    df = encode_categorical_features(df)
    
    # Step 6: Scale features if requested
    if should_scale_features:
        df = scale_features(df)
    
    logger.info("Preprocessing pipeline completed")
    logger.info(f"Final shape: {df.shape}")
    
    return df