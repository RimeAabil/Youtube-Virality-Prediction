import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import joblib

class FeatureExpert:
    """Senior ML Feature Engineering for YouTube Virality."""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.power_words = ["secret", "revealed", "stop", "never", "best", "viral", "amazing", "hack", "top"]
        
    def convert_duration_to_seconds(self, duration):
        """Convert ISO 8601 duration to seconds."""
        if not duration: return 0
        duration = str(duration).replace('PT', '')
        hours = re.search(r'(\d+)H', duration)
        minutes = re.search(r'(\d+)M', duration)
        seconds = re.search(r'(\d+)S', duration)
        total = 0
        if hours: total += int(hours.group(1)) * 3600
        if minutes: total += int(minutes.group(1)) * 60
        if seconds: total += int(seconds.group(1))
        return total

    def compute_legacy_engagement(self, views, likes, comments):
        """Legacy engagement score: (likes + comments * 2) / views."""
        if views == 0: return 0
        return (likes + comments * 2) / views

    def compute_virality_score(self, df):
        """
        Enhanced Virality Score combining multiple engagement signals.
        
        VS = (Engagement Rate × Velocity Factor × Reach Amplification) / Channel Baseline
        
        Components:
        - Engagement Rate: weighted combination of likes, comments (comments worth 3x)
        - Velocity Factor: views per day since publication
        - Reach Amplification: how far beyond subscriber base the video reached
        - Engagement Density: engagement per minute of content
        """
        
        # Prevent division by zero
        df['sub_count_safe'] = df['subscriber_count'].apply(lambda x: max(x, 100))
        df['duration_safe'] = df['duration_seconds'].apply(lambda x: max(x, 1))
        
        # 1. Engagement Rate (weighted by interaction depth)
        df['engagement_rate'] = (
            (df['like_count'] * 1.0 + 
             df['comment_count'] * 3.0) / df['view_count'].apply(lambda x: max(x, 1))
        )
        
        # 2. Velocity Factor (views per day)
        # Ensure both datetimes are timezone-naive for comparison
        now = pd.Timestamp.now().tz_localize(None)
        df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)
        df['days_since_publish'] = (now - df['published_at']).dt.total_seconds() / 86400
        df['days_since_publish'] = df['days_since_publish'].apply(lambda x: max(x, 1))
        df['velocity'] = df['view_count'] / df['days_since_publish']
        
        # 3. Reach Amplification (how far beyond subscriber base)
        df['reach_ratio'] = df['view_count'] / df['sub_count_safe']
        
        # 4. Engagement Density (engagement per minute of content)
        df['engagement_density'] = (
            (df['like_count'] + df['comment_count']) / 
            (df['duration_safe'] / 60)  # per minute of video
        )
        
        # 5. Combined Virality Score (weighted components)
        df['virality_score_raw'] = (
            df['engagement_rate'] * 0.25 +
            np.log1p(df['velocity']) * 0.30 +
            np.log1p(df['reach_ratio']) * 0.30 +
            np.log1p(df['engagement_density']) * 0.15
        )
        
        # Normalize to 0-100 scale
        score_min = df['virality_score_raw'].min()
        score_max = df['virality_score_raw'].max()
        
        if score_max > score_min:
            df['virality_score'] = (
                (df['virality_score_raw'] - score_min) / 
                (score_max - score_min) * 100
            )
        else:
            df['virality_score'] = 50.0  # Default if all scores are the same
        
        # Binary label: Top 10% are viral
        threshold = df['virality_score'].quantile(0.9)
        df['is_viral_label'] = (df['virality_score'] >= threshold).astype(int)
        
        return df

    def extract_nlp_features(self, title):
        """Advanced NLP features for the video title."""
        title_str = str(title)
        title_clean = title_str.lower()
        
        features = {
            'title_len': len(title_str),
            'title_length': len(title_str),
            'title_sentiment': self.analyzer.polarity_scores(title_str)['compound'],
            'emoji_count': len(re.findall(r'[^\w\s,]', title_str)),
            'contains_emojis': 1 if len(re.findall(r'[^\w\s,]', title_str)) > 0 else 0,
            'is_uppercase': 1 if any(w.isupper() for w in title_str.split()) else 0,
            'has_power_word': 1 if any(pw in title_clean for pw in self.power_words) else 0,
            'exclamation_count': title_str.count('!'),
            'contains_exclamation': 1 if '!' in title_str else 0,
            'question_count': title_str.count('?'),
            'all_caps_word_count': len([w for w in title_str.split() if w.isupper() and len(w) > 1]),
            'contains_number': 1 if any(char.isdigit() for char in title_str) else 0
        }
        return features

    def get_time_features(self, dt):
        """Cyclical encoding for hours and days + weekend flag."""
        hour = dt.hour
        weekday = dt.weekday()
        return {
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * weekday / 7),
            'day_cos': np.cos(2 * np.pi * weekday / 7),
            'is_weekend': 1 if weekday >= 5 else 0
        }
    
    def prepare_dataset(self, df_videos, df_stats, df_channels):
        """Full pipeline for dataset preparation with holistic features."""
        # Merge stats and take latest
        stats_latest = df_stats.sort_values('captured_at').groupby('video_id').last().reset_index()
        
        # Merge dataframes
        df = df_videos.merge(stats_latest, on='video_id', how='inner')
        df = df.merge(df_channels, on='channel_id', suffixes=('', '_chan'), how='inner')
        
        # 1. Holistic Preprocessing
        df['duration_seconds'] = df['duration'].apply(self.convert_duration_to_seconds)
        df['legacy_engagement_score'] = df.apply(
            lambda x: self.compute_legacy_engagement(x['view_count'], x['like_count'], x['comment_count']), 
            axis=1
        )
        
        # Convert published_at to datetime (will be made timezone-naive in compute_virality_score)
        df['published_at'] = pd.to_datetime(df['published_at'])
        
        # 2. Compute enhanced virality score and labels
        df = self.compute_virality_score(df)
        
        # 3. Extract NLP features
        nlp_data = df['title'].apply(self.extract_nlp_features).apply(pd.Series)
        
        # 4. Extract Time features (published_at is now timezone-naive)
        time_data = df['published_at'].apply(self.get_time_features).apply(pd.Series)
        
        # 5. Additional engineered features
        df['subscriber_count_log'] = np.log1p(df['subscriber_count'])
        df['view_count_log'] = np.log1p(df['view_count'])
        df['like_count_log'] = np.log1p(df['like_count'])
        df['comment_count_log'] = np.log1p(df['comment_count'])
        
        # Final Assemblage
        df = pd.concat([df, nlp_data, time_data], axis=1)
        
        return df