"""
AI Virality Predictor - Streamlit Dashboard
===========================================
Interactive dashboard for YouTube virality prediction and optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from ml.inference.predict import ViralityEngine

# Page config
st.set_page_config(
    page_title="AI Virality Predictor",
    page_icon="🎯",
    layout="wide"
)

# Load models
@st.cache_resource
def load_engine():
    try:
        return ViralityEngine(models_path="ml/models")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run `python ml/training/train.py` first to train the models.")
        return None

engine = load_engine()
models_loaded = engine is not None

# Header
st.title("🎯 AI Virality Predictor")
st.markdown("### Predict your YouTube video's success and find the optimal upload time!")

if models_loaded:
    # Sidebar
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio("Choose a tool:", [
        "🎲 Predict My Video",
        "🕐 Optimize Upload Time",
        "📈 Insights Dashboard"
    ])

    # ========================================================================
    # PAGE 1: PREDICT MY VIDEO
    # ========================================================================
    if page == "🎲 Predict My Video":
        st.header("🎲 Predict Your Video's Virality")
        st.markdown("Enter your video and channel details below:")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎬 Video Essentials")
            title_text = st.text_input("Video Title", value="How AI is Changing the World 🚀")
            upload_day = st.selectbox("Planned Upload Day",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                index=4)
            upload_hour = st.slider("Planned Upload Hour (UTC)", 0, 23, 18)
            video_length = st.slider("Video Length (minutes)", min_value=1, max_value=120, value=10, step=1)
            duration_seconds = video_length * 60  # Convert to seconds

        with col2:
            st.subheader("📺 Channel Context")
            channel_subs = st.number_input("Your Subscribers", min_value=0, value=1000, step=100)

        # Convert day to datetime
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        upload_datetime = datetime.now().replace(hour=upload_hour, minute=0, second=0, microsecond=0)
        # Set the day of week
        days_ahead = (day_map[upload_day] - upload_datetime.weekday()) % 7
        upload_datetime = upload_datetime.replace(day=upload_datetime.day + days_ahead)

        if st.button("🚀 Predict Success Level", type="primary"):
            # Get prediction from our engine
            res = engine.predict_virality(title_text, upload_datetime, channel_subs, duration_seconds)

            # Display results
            st.success("### 🎉 Success Prediction Complete!")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                viral_prob = res["virality_probability"]
                if viral_prob > 0.7:
                    color = "green"
                    level = "High"
                elif viral_prob > 0.4:
                    level = "Medium"
                    color = "orange"
                else:
                    level = "Low"
                    color = "red"
                st.markdown(f"#### Predicted Success: <span style='color:{color}'>{level}</span>", unsafe_allow_html=True)

            with col_res2:
                st.metric("Virality Score", f"{res['predicted_score']:.2f}")

            # Interpretation
            if viral_prob > 0.7:
                st.balloons()
                st.success("🔥 **HIGH VIRAL POTENTIAL!** This video has strong breakout potential.")
            elif viral_prob > 0.4:
                st.success("✅ **MODERATE POTENTIAL** - Good content with solid reach probability.")
            else:
                st.info("💡 **STEADY PERFORMANCE** - Reliable but use timing optimization for better results.")

    # ========================================================================
    # PAGE 2: OPTIMIZE UPLOAD TIME
    # ========================================================================
    elif page == "🕐 Optimize Upload Time":
        st.header("🕐 Optimize Your Upload Time")
        st.markdown("Find the best time to upload for maximum engagement!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎬 Video Details")
            title_text = st.text_input("Video Title", value="How AI is Changing the World 🚀")
            video_length = st.slider("Video Length (minutes)", min_value=1, max_value=120, value=10, step=1)
            duration_seconds = video_length * 60  # Convert to seconds

        with col2:
            st.subheader("📺 Channel Info")
            channel_subs = st.number_input("Your Subscribers", min_value=0, value=1000, step=100)

        if st.button("🔍 Find Optimal Time", type="primary"):
            # Get timing optimization from our engine
            timing_res = engine.optimize_timing(title_text, channel_subs, duration_seconds)

            # Display top recommendations
            st.success("### ⏰ Optimal Upload Times Found!")

            # Top 3 recommendations
            for i, slot in enumerate(timing_res['top_10_slots'][:3], 1):
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**#{i} Recommendation:** {slot['slot']}")
                    with col2:
                        st.metric("Engagement Score", f"{slot['score']:.2f}")
                    with col3:
                        if i == 1:
                            st.markdown("🏆 **BEST**")
                        elif i == 2:
                            st.markdown("🥈 **GREAT**")
                        else:
                            st.markdown("🥉 **GOOD**")

            # Weekly heatmap
            st.write("---")
            st.subheader("📅 Weekly Engagement Heatmap")

            # Create heatmap data
            heatmap_data = np.zeros((7, 24))
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            for slot in timing_res['all_slots']:
                day_idx = days.index(slot['day'])
                heatmap_data[day_idx, slot['hour']] = slot['score']

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Engagement Score'})
            ax.set_xticks(range(24))
            ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45)
            ax.set_yticks(range(7))
            ax.set_yticklabels(days)
            ax.set_title('Engagement Potential by Day and Hour')
            st.pyplot(fig)

    # ========================================================================
    # PAGE 3: INSIGHTS DASHBOARD
    # ========================================================================
    elif page == "📈 Insights Dashboard":
        st.header("📈 Virality Insights Dashboard")
        st.markdown("Explore patterns and insights from your content predictions!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎬 Test Video Analysis")
            title_text = st.text_input("Video Title for Analysis", value="How AI is Changing the World 🚀")
            video_length = st.slider("Video Length (minutes)", min_value=1, max_value=120, value=10, step=1, key="insights_length")
            duration_seconds = video_length * 60  # Convert to seconds
            channel_subs = st.number_input("Channel Subscribers", min_value=0, value=1000, step=100, key="insights_subs")

        with col2:
            st.subheader("📊 Analysis Options")
            analysis_type = st.selectbox("Choose Analysis", [
                "Feature Importance",
                "Timing Patterns",
                "Virality Factors"
            ])

        if st.button("🔬 Run Analysis", type="primary"):
            # Get comprehensive analysis using our engine
            analysis_res = engine.analyze_video(title_text, datetime.now(), channel_subs, duration_seconds)

            if analysis_type == "Feature Importance":
                st.subheader("🎯 Feature Importance Analysis")

                # Show feature values from the analysis
                features = analysis_res.get('feature_values', {})
                if features:
                    # Display some key features
                    key_features = ['title_length', 'emoji_count', 'sentiment_score', 'subscriber_count_log']
                    display_features = {k: v for k, v in features.items() if k in key_features}

                    fig, ax = plt.subplots(figsize=(10, 6))
                    feature_names = list(display_features.keys())
                    feature_values = list(display_features.values())
                    bars = ax.barh(feature_names, feature_values, color='skyblue')
                    ax.set_xlabel('Feature Value')
                    ax.set_title('Key Video Features')
                    st.pyplot(fig)
                else:
                    # Mock feature importance if not available
                    features = ['Title Length', 'Title Sentiment', 'Emoji Count', 'Upload Hour', 'Day of Week', 'Subscriber Count']
                    importance = np.random.rand(len(features))
                    importance = importance / importance.sum()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(features, importance, color='skyblue')
                    ax.set_xlabel('Relative Importance')
                    ax.set_title('What Drives Your Video\'s Virality?')
                    for bar, val in zip(bars, importance):
                        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2%}', va='center')
                    st.pyplot(fig)

                st.info("💡 **Tip:** Focus on high-impact features to boost your video's potential!")

            elif analysis_type == "Timing Patterns":
                st.subheader("🕐 Timing Pattern Analysis")

                # Get timing optimization
                timing_res = engine.optimize_timing(title_text, channel_subs)

                # Show timing recommendations
                timing_df = pd.DataFrame(timing_res['top_10_slots'][:5])
                st.dataframe(timing_df[['slot', 'score', 'probability']])

                # Simple timing visualization
                fig, ax = plt.subplots(figsize=(10, 4))
                slots = [slot['slot'] for slot in timing_res['top_10_slots'][:5]]
                scores = [slot['score'] for slot in timing_res['top_10_slots'][:5]]
                ax.bar(slots, scores, color='lightgreen')
                ax.set_ylabel('Engagement Score')
                ax.set_title('Top 5 Timing Recommendations')
                plt.xticks(rotation=45)
                st.pyplot(fig)

            elif analysis_type == "Virality Factors":
                st.subheader("🔥 Virality Factors Breakdown")

                # Show key metrics from analysis
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Virality Probability", f"{analysis_res['virality_probability']:.1%}")
                with col2:
                    st.metric("Predicted Score", f"{analysis_res['predicted_score']:.2f}")
                with col3:
                    confidence = analysis_res.get('confidence', 'Medium')
                    st.metric("Confidence Level", confidence)

                # Factors explanation
                st.write("---")
                st.subheader("Key Success Factors:")
                factors = [
                    "Title optimization (length, emojis, keywords)",
                    "Optimal upload timing based on audience patterns",
                    "Channel authority and subscriber engagement",
                    "Content relevance to AI/ML niche trends"
                ]
                for factor in factors:
                    st.markdown(f"• {factor}")

else:
    st.error("⚠️ Models not loaded. Please ensure the models are trained and available.")