import os
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()

class YouTubeIngestor:
    """Production Data Acquisition Layer for YouTube."""
    
    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY not found in environment variables. Please set it in .env file")
        
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.raw_path = "data/raw"
        os.makedirs(self.raw_path, exist_ok=True)
        
    def collect_niche_data(self, queries=None, max_per_query=50):
        """
        Collect balanced dataset: viral and standard videos from the niche.
        
        Args:
            queries: List of search queries (default: ML/AI topics)
            max_per_query: Maximum results per query
            
        Returns:
            dict: Summary statistics of collection
        """
        if queries is None:
            queries = [
                "Machine Learning Tutorial",
                "Artificial Intelligence",
                "Data Science",
                "Deep Learning",
                "Python Programming",
                "AI News"
            ]
        
        all_video_ids = []
        channel_ids = set()
        
        print("="*60)
        print("YOUTUBE DATA COLLECTION")
        print("="*60)
        
        for query in queries:
            print(f"\n🔍 Searching: '{query}'")
            
            try:
                # 1. Fetch Viral Videos (Order by ViewCount)
                print("  → Fetching viral content...")
                viral = self._search(query, 'viewCount', max_per_query)
                viral_count = self._process_search_results(viral, all_video_ids, channel_ids)
                
                # 2. Fetch Recent/Standard Videos (Order by Date)
                print("  → Fetching recent content...")
                recent = self._search(query, 'date', max_per_query)
                recent_count = self._process_search_results(recent, all_video_ids, channel_ids)
                
                print(f"  ✓ Found {viral_count} viral + {recent_count} recent videos")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"  ✗ Error fetching '{query}': {e}")
                continue
        
        total_videos = len(all_video_ids)
        total_channels = len(channel_ids)
        
        print(f"\n📊 Collection Summary:")
        print(f"  Unique Videos:   {total_videos}")
        print(f"  Unique Channels: {total_channels}")
        
        if total_videos == 0:
            print("\n⚠️ No videos collected. Check your API key and quota.")
            return {'videos': 0, 'channels': 0, 'statistics': 0}
        
        # Hydrate detailed data
        print(f"\n📥 Fetching detailed video data...")
        videos, stats = self._get_video_details(all_video_ids)
        
        print(f"📥 Fetching channel data...")
        channels = self._get_channel_details(list(channel_ids))
        
        # Save to CSV
        print(f"\n💾 Saving data...")
        pd.DataFrame(videos).to_csv(f"{self.raw_path}/videos.csv", index=False)
        pd.DataFrame(stats).to_csv(f"{self.raw_path}/statistics.csv", index=False)
        pd.DataFrame(channels).to_csv(f"{self.raw_path}/channels.csv", index=False)
        
        print(f"✓ Saved to {self.raw_path}/")
        
        # Summary statistics
        df_videos = pd.DataFrame(videos)
        df_stats = pd.DataFrame(stats)
        
        print("\n" + "="*60)
        print("DATA COLLECTION COMPLETE")
        print("="*60)
        print(f"Videos collected:     {len(videos)}")
        print(f"Statistics records:   {len(stats)}")
        print(f"Channels collected:   {len(channels)}")
        
        if len(stats) > 0:
            print(f"\nView count range:     {df_stats['view_count'].min():,} - {df_stats['view_count'].max():,}")
            print(f"Average views:        {df_stats['view_count'].mean():,.0f}")
            print(f"Average likes:        {df_stats['like_count'].mean():,.0f}")
            print(f"Average comments:     {df_stats['comment_count'].mean():,.0f}")
        
        return {
            'videos': len(videos),
            'channels': len(channels),
            'statistics': len(stats)
        }

    def _process_search_results(self, results, video_ids, channel_ids):
        """Process search results and update tracking sets."""
        count = 0
        for item in results.get('items', []):
            if 'videoId' in item['id']:
                v_id = item['id']['videoId']
                if v_id not in video_ids:
                    video_ids.append(v_id)
                    channel_ids.add(item['snippet']['channelId'])
                    count += 1
        return count

    def _search(self, query, order, max_results):
        """Execute YouTube search API call."""
        params = {
            'part': 'snippet',
            'q': query,
            'type': 'video',
            'relevanceLanguage': 'en',
            'order': order,
            'maxResults': min(max_results, 50),  # API limit is 50
            'key': self.api_key,
            'publishedAfter': '2023-01-01T00:00:00Z',  # Focus on recent content
            'videoDuration': 'medium'  # Prefer 4-20 minute videos
        }
        
        response = requests.get(f"{self.base_url}/search", params=params)
        response.raise_for_status()
        return response.json()

    def _get_video_details(self, video_ids):
        """Fetch detailed video information in batches."""
        v_data, s_data = [], []
        total_batches = (len(video_ids) + 49) // 50
        
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            batch_num = (i // 50) + 1
            
            print(f"  → Batch {batch_num}/{total_batches} ({len(batch)} videos)")
            
            try:
                params = {
                    'part': 'snippet,statistics,contentDetails',
                    'id': ','.join(batch),
                    'key': self.api_key
                }
                
                response = requests.get(f"{self.base_url}/videos", params=params)
                response.raise_for_status()
                res = response.json()
                
                for item in res.get('items', []):
                    # Video metadata
                    v_data.append({
                        'video_id': item['id'],
                        'channel_id': item['snippet']['channelId'],
                        'title': item['snippet']['title'],
                        'description': item['snippet'].get('description', '')[:500],  # Truncate long descriptions
                        'published_at': item['snippet']['publishedAt'],
                        'tags': ','.join(item['snippet'].get('tags', [])[:10]),  # Limit tags
                        'duration': item.get('contentDetails', {}).get('duration', ''),
                        'category_id': item['snippet'].get('categoryId', '')
                    })
                    
                    # Statistics snapshot
                    s_data.append({
                        'video_id': item['id'],
                        'view_count': int(item['statistics'].get('viewCount', 0)),
                        'like_count': int(item['statistics'].get('likeCount', 0)),
                        'comment_count': int(item['statistics'].get('commentCount', 0)),
                        'captured_at': datetime.now().isoformat()
                    })
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ✗ Error in batch {batch_num}: {e}")
                continue
        
        return v_data, s_data

    def _get_channel_details(self, channel_ids):
        """Fetch channel information in batches."""
        c_data = []
        total_batches = (len(channel_ids) + 49) // 50
        
        for i in range(0, len(channel_ids), 50):
            batch = channel_ids[i:i+50]
            batch_num = (i // 50) + 1
            
            print(f"  → Batch {batch_num}/{total_batches} ({len(batch)} channels)")
            
            try:
                params = {
                    'part': 'snippet,statistics',
                    'id': ','.join(batch),
                    'key': self.api_key
                }
                
                response = requests.get(f"{self.base_url}/channels", params=params)
                response.raise_for_status()
                res = response.json()
                
                for item in res.get('items', []):
                    c_data.append({
                        'channel_id': item['id'],
                        'channel_title': item['snippet']['title'],
                        'subscriber_count': int(item['statistics'].get('subscriberCount', 0)),
                        'video_count': int(item['statistics'].get('videoCount', 0)),
                        'view_count_total': int(item['statistics'].get('viewCount', 0)),
                        'created_at': item['snippet'].get('publishedAt', '')
                    })
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ✗ Error in batch {batch_num}: {e}")
                continue
        
        return c_data

    def update_statistics(self, video_ids=None):
        """
        Update statistics for existing videos (for tracking over time).
        
        Args:
            video_ids: List of video IDs to update, or None to update all from videos.csv
        """
        if video_ids is None:
            # Load existing videos
            videos_path = f"{self.raw_path}/videos.csv"
            if not os.path.exists(videos_path):
                print("No existing videos.csv found. Run collect_niche_data() first.")
                return
            
            df_videos = pd.read_csv(videos_path)
            video_ids = df_videos['video_id'].tolist()
        
        print(f"\n🔄 Updating statistics for {len(video_ids)} videos...")
        
        _, new_stats = self._get_video_details(video_ids)
        
        # Append to existing statistics
        stats_path = f"{self.raw_path}/statistics.csv"
        if os.path.exists(stats_path):
            existing_stats = pd.read_csv(stats_path)
            combined_stats = pd.concat([existing_stats, pd.DataFrame(new_stats)], ignore_index=True)
            combined_stats.to_csv(stats_path, index=False)
            print(f"✓ Appended {len(new_stats)} new statistics records")
        else:
            pd.DataFrame(new_stats).to_csv(stats_path, index=False)
            print(f"✓ Created statistics.csv with {len(new_stats)} records")


def main():
    """Main execution function."""
    try:
        ingestor = YouTubeIngestor()
        
        # Collect initial dataset
        result = ingestor.collect_niche_data(
            queries=[
                "Machine Learning Tutorial",
                "Artificial Intelligence Explained",
                "Data Science Projects",
                "Deep Learning",
                "Python Programming",
                "AI News Today"
            ],
            max_per_query=30  # Adjust based on your API quota
        )
        
        print(f"\n✅ Data collection completed successfully!")
        print(f"   You can now run train.py to train the models.")
        
    except Exception as e:
        print(f"\n❌ Error during data collection: {e}")
        print(f"   Please check your API key and internet connection.")


if __name__ == "__main__":
    main()