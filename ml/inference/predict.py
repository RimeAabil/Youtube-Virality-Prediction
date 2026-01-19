import joblib
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path to import FeatureExpert
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))
from features import FeatureExpert

class ViralityEngine:
    """Production Prediction Engine for YouTube AI Virality Platform."""
    
    def __init__(self, models_path="ml/models"):
        """Initialize the prediction engine with trained models."""
        try:
            self.clf = joblib.load(f"{models_path}/virality_classifier_v2.joblib")
            self.reg = joblib.load(f"{models_path}/timing_regressor_v2.joblib")
            self.feature_cols = joblib.load(f"{models_path}/features_metadata.joblib")
            self.fe = FeatureExpert()  # Create new instance instead of loading
            print("✓ Models loaded successfully")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Models not found. Please run train.py first to train the models. Error: {e}"
            )
        
    def predict_virality(self, title, published_at=None, subscriber_count=1000, 
                        duration_seconds=600, view_count=0, like_count=0, comment_count=0):
        """
        Analyze a video and return comprehensive virality prediction.
        
        Args:
            title: Video title string
            published_at: datetime object or None (uses current time)
            subscriber_count: Channel subscriber count
            duration_seconds: Video duration in seconds (default 10 min)
            view_count: Current view count (for existing videos)
            like_count: Current like count (for existing videos)
            comment_count: Current comment count (for existing videos)
            
        Returns:
            dict: Comprehensive virality analysis
        """
        if published_at is None:
            published_at = datetime.now()
        elif isinstance(published_at, str):
            published_at = pd.to_datetime(published_at)
            
        # 1. Extract NLP Features
        nlp_feats = self.fe.extract_nlp_features(title)
        
        # 2. Extract Time Features
        time_feats = self.fe.get_time_features(published_at)
        
        # 3. Calculate engagement metrics
        engagement_rate = 0
        velocity = 0
        reach_ratio = 0
        engagement_density = 0
        
        if view_count > 0:
            engagement_rate = (like_count * 1.0 + comment_count * 3.0) / max(view_count, 1)
            days_since = max((datetime.now() - published_at).total_seconds() / 86400, 1)
            velocity = view_count / days_since
            reach_ratio = view_count / max(subscriber_count, 100)
            engagement_density = (like_count + comment_count) / max(duration_seconds / 60, 1)
        
        # 4. Prepare feature vector
        current_features = {
            **nlp_feats,
            **time_feats,
            'subscriber_count_log': np.log1p(subscriber_count),
            'duration_seconds': duration_seconds,
            'legacy_engagement_score': self.fe.compute_legacy_engagement(view_count, like_count, comment_count),
            'engagement_rate': engagement_rate,
            'velocity': velocity,
            'reach_ratio': reach_ratio,
            'engagement_density': engagement_density
        }
        
        # Ensure all required features are present
        for col in self.feature_cols:
            if col not in current_features:
                current_features[col] = 0
        
        # 5. Predict Virality Probability
        df_input = pd.DataFrame([current_features])
        X = df_input[self.feature_cols]
        
        virality_prob = self.clf.predict_proba(X)[0][1]
        is_viral = self.clf.predict(X)[0]
        
        # 6. Predict Virality Score
        predicted_score = self.reg.predict(X)[0]
        
        return {
            'virality_probability': float(virality_prob),
            'is_viral': bool(is_viral),
            'predicted_score': float(predicted_score),
            'interpretation': self._interpret_score(virality_prob, predicted_score),
            'confidence': self._get_confidence(virality_prob),
            'feature_values': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                              for k, v in current_features.items()},
            'feature_importance': self._get_top_features()
        }
    
    def optimize_timing(self, title, subscriber_count=1000, duration_seconds=600, days_ahead=7):
        """
        Find optimal publishing times for the next N days.
        
        Args:
            title: Video title string
            subscriber_count: Channel subscriber count
            duration_seconds: Video duration in seconds
            days_ahead: Number of days to analyze (default 7)
            
        Returns:
            dict: Timing optimization results
        """
        # Extract static features
        nlp_feats = self.fe.extract_nlp_features(title)
        
        timing_results = []
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        current_date = datetime.now()
        
        for day_offset in range(days_ahead):
            target_date = current_date + timedelta(days=day_offset)
            day_index = target_date.weekday()
            
            for hour in range(24):
                # Create datetime for this slot
                slot_datetime = target_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                
                # Extract time features for this slot
                time_feats = self.fe.get_time_features(slot_datetime)
                
                # Prepare feature vector
                slot_features = {
                    **nlp_feats,
                    **time_feats,
                    'subscriber_count_log': np.log1p(subscriber_count),
                    'duration_seconds': duration_seconds,
                    'legacy_engagement_score': 0,
                    'engagement_rate': 0,
                    'velocity': 0,
                    'reach_ratio': 0,
                    'engagement_density': 0
                }
                
                # Ensure all features present
                for col in self.feature_cols:
                    if col not in slot_features:
                        slot_features[col] = 0
                
                # Predict score for this slot
                X_slot = pd.DataFrame([slot_features])[self.feature_cols]
                pred_score = self.reg.predict(X_slot)[0]
                pred_prob = self.clf.predict_proba(X_slot)[0][1]
                
                timing_results.append({
                    'datetime': slot_datetime.isoformat(),
                    'day': days[day_index],
                    'hour': hour,
                    'slot': f"{days[day_index]} {hour:02d}:00",
                    'score': float(pred_score),
                    'probability': float(pred_prob),
                    'is_weekend': day_index >= 5
                })
        
        # Sort by score
        timing_results = sorted(timing_results, key=lambda x: x['score'], reverse=True)
        
        return {
            'best_time': timing_results[0],
            'top_10_slots': timing_results[:10],
            'all_slots': timing_results,
            'summary': self._generate_timing_summary(timing_results)
        }
    
    def analyze_video(self, title, published_at=None, subscriber_count=1000, 
                     duration_seconds=600, view_count=0, like_count=0, comment_count=0):
        """
        Complete video analysis combining virality prediction and timing optimization.
        
        Returns:
            dict: Comprehensive analysis report
        """
        # Get virality prediction
        virality = self.predict_virality(
            title, published_at, subscriber_count, duration_seconds,
            view_count, like_count, comment_count
        )
        
        # Get timing optimization (only if it's a future video)
        timing = None
        if view_count == 0 or (published_at and published_at > datetime.now()):
            timing = self.optimize_timing(title, subscriber_count, duration_seconds)
        
        return {
            'title': title,
            'analysis_timestamp': datetime.now().isoformat(),
            'virality_prediction': virality,
            'timing_optimization': timing,
            'recommendations': self._generate_recommendations(virality, timing)
        }
    
    def _interpret_score(self, prob, score):
        """Generate human-readable interpretation."""
        if prob > 0.8:
            return f"🔥 High Virality Potential (Score: {score:.1f}/100)"
        elif prob > 0.6:
            return f"📈 Above Average Potential (Score: {score:.1f}/100)"
        elif prob > 0.4:
            return f"⚖️ Moderate Potential (Score: {score:.1f}/100)"
        else:
            return f"📊 Standard Content (Score: {score:.1f}/100)"
    
    def _get_confidence(self, prob):
        """Calculate prediction confidence."""
        # Distance from 0.5 indicates confidence
        confidence = abs(prob - 0.5) * 2
        if confidence > 0.7:
            return "High"
        elif confidence > 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _get_top_features(self, n=5):
        """Get top N most important features."""
        importances = self.clf.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False).head(n)
        
        return feature_imp.to_dict('records')
    
    def _generate_timing_summary(self, timing_results):
        """Generate summary insights from timing analysis."""
        # Best day of week
        day_scores = {}
        for result in timing_results:
            day = result['day']
            if day not in day_scores:
                day_scores[day] = []
            day_scores[day].append(result['score'])
        
        best_day = max(day_scores.keys(), key=lambda d: np.mean(day_scores[d]))
        
        # Best time of day
        hour_scores = {}
        for result in timing_results:
            hour = result['hour']
            if hour not in hour_scores:
                hour_scores[hour] = []
            hour_scores[hour].append(result['score'])
        
        best_hour = max(hour_scores.keys(), key=lambda h: np.mean(hour_scores[h]))
        
        return {
            'best_day': best_day,
            'best_hour': f"{best_hour:02d}:00",
            'weekend_better': np.mean([r['score'] for r in timing_results if r['is_weekend']]) >
                             np.mean([r['score'] for r in timing_results if not r['is_weekend']])
        }
    
    def _generate_recommendations(self, virality, timing):
        """Generate actionable recommendations."""
        recommendations = []
        
        # Based on virality probability
        prob = virality['virality_probability']
        if prob > 0.7:
            recommendations.append("✅ Strong viral potential detected. Consider promoting this content.")
        elif prob < 0.3:
            recommendations.append("⚠️ Consider optimizing title for better engagement.")
        
        # Based on features
        features = virality['feature_values']
        if features.get('title_sentiment', 0) < 0:
            recommendations.append("💡 Negative sentiment detected. Consider more positive framing.")
        if features.get('has_power_word', 0) == 0:
            recommendations.append("💡 Add power words like 'secret', 'revealed', or 'amazing' to boost engagement.")
        if features.get('contains_number', 0) == 0:
            recommendations.append("💡 Numbers in titles tend to perform better (e.g., '10 Tips', '5 Ways').")
        
        # Based on timing
        if timing:
            best_time = timing['best_time']
            recommendations.append(f"⏰ Optimal publishing time: {best_time['slot']}")
        
        return recommendations


def main():
    """Example usage of the ViralityEngine."""
    engine = ViralityEngine()
    
    # Example 1: Analyze a new video idea
    print("="*60)
    print("EXAMPLE 1: New Video Analysis")
    print("="*60)
    
    result = engine.analyze_video(
        title="10 Amazing Machine Learning Secrets Revealed!",
        subscriber_count=50000,
        duration_seconds=720
    )
    
    print(f"\nTitle: {result['title']}")
    print(f"Virality Probability: {result['virality_prediction']['virality_probability']:.2%}")
    print(f"Interpretation: {result['virality_prediction']['interpretation']}")
    print(f"Confidence: {result['virality_prediction']['confidence']}")
    
    if result['timing_optimization']:
        best = result['timing_optimization']['best_time']
        print(f"\nBest Time to Publish: {best['slot']}")
        print(f"Predicted Score: {best['score']:.2f}")
    
    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  {rec}")


if __name__ == "__main__":
    main()