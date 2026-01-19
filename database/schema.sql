-- Refined Schema for YouTube AI Virality Platform V2

CREATE TABLE IF NOT EXISTS channels (
    channel_id VARCHAR(50) PRIMARY KEY,
    channel_title TEXT,
    subscriber_count INT,
    video_count INT,
    view_count_total BIGINT,
    avg_views_last_10_videos FLOAT,
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS videos (
    video_id VARCHAR(50) PRIMARY KEY,
    channel_id VARCHAR(50) REFERENCES channels(channel_id),
    title TEXT,
    description TEXT,
    published_at TIMESTAMP,
    tags TEXT,
    duration_iso TEXT,
    is_viral_manual BOOLEAN, -- Used for ground truth if provided
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS video_stats_daily (
    stat_entry_id SERIAL PRIMARY KEY,
    video_id VARCHAR(50) REFERENCES videos(video_id),
    view_count BIGINT,
    like_count INT,
    comment_count INT,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS features_ml (
    feature_id SERIAL PRIMARY KEY,
    video_id VARCHAR(50) REFERENCES videos(video_id),
    title_length INT,
    title_sentiment FLOAT,
    emoji_count INT,
    contains_emojis BOOLEAN,
    is_uppercase BOOLEAN,
    has_power_word BOOLEAN,
    exclamation_count INT,
    contains_exclamation BOOLEAN,
    question_count INT,
    all_caps_word_count INT,
    contains_number BOOLEAN,
    hour_sin FLOAT,
    hour_cos FLOAT,
    day_of_week INT,
    is_weekend BOOLEAN,
    duration_seconds INT,
    legacy_engagement_score FLOAT,
    subscriber_count_log FLOAT,
    virality_score FLOAT, -- Target variable
    is_viral_label INT -- Binary target
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    input_title TEXT,
    predicted_virality_score FLOAT,
    is_viral_predicted BOOLEAN,
    best_hour_recommended INT,
    best_day_recommended VARCHAR(10),
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
