# YouTube Virality Prediction Project

This project implements a pipeline to collect YouTube data, preprocess it, engineer features, and predict video virality using a sophisticated "Virality Score".

## Project Structure

### Core Scripts
*   **`youtube_api_config.py`**: YouTube Data API v3 configuration and utility functions. Handles authentication (API Key/OAuth), video searching, comment retrieval, and data masking.
*   **`data_preprocessing.py`**: Initial raw data processing. Extracts structured features from API responses (videos, comments, channels) and handles basic data cleaning.
*   **`advanced_preprocessing.py`**: Advanced feature engineering pipeline. Features include:
    *   **Virality Score**: Calculates engagement, velocity, and reach metrics.
    *   **Text Features**: Title length, emoji counts, clickbait detection.
    *   **Temporal Features**: Best upload times, weekend vs weekday analysis.
*   **`youtube_eda.py`**: Exploratory Data Analysis module. Contains the **Enhanced Virality Score V2** implementation (`compute_virality_score_v2`) and various plotting functions.
*   **`task2.py`**: Main pipeline orchestration script. Integrates data loading, preprocessing, feature engineering, and report generation.

### Notebooks
*   `Task1.ipynb`: Data collection and initial exploration.
*   `Task2.ipynb`: Pipeline development and experimentation.

## Key Features

### Enhanced Virality Score V2
Located in `youtube_eda.py`, this new metric calculates a comprehensive score based on:
1.  **Engagement Rate**: Weighted likes, comments, and views.
2.  **Velocity Factor**: Views per day since publication.
3.  **Reach Amplification**: Views-to-subscribers ratio.
4.  **Retention Quality**: Engagement density per minute.

## Setup & Usage

### Prerequisites
*   Python 3.8+
*   YouTube Data API Key (or OAuth credentials)

### Installation
Install the required dependencies:
```bash
pip install pandas numpy seaborn matplotlib plotly wordcloud google-api-python-client google-auth-oauthlib google-auth-httplib2
```

### Running the Pipeline
To run the main analysis pipeline:
```bash
python "task2.py"
```

To run the custom virality score test:
```bash
python test_virality_score.py
```

## Configuration
Ensure your `.env` file or environment variables contain:
*   `YOUTUBE_API_KEY`: Your Google Cloud API Key for YouTube Data API v3.