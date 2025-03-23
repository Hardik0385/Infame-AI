import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import tweepy
import praw
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from apscheduler.schedulers.background import BackgroundScheduler
import redis
from transformers import pipeline, AutoModelForCausalLM

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("influenceiq.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize Redis for caching
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD', None),
        decode_responses=True
    )
    redis_client.ping()  # Test connection
    logger.info("Redis connection established")
except redis.ConnectionError:
    logger.warning("Redis server not available. Caching will be disabled.")
    redis_client = None

# Initialize NLTK components
nltk.download('vader_lexicon')
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize DeepSeek AI model
deepseek_model = None
deepseek_pipeline = None

def initialize_deepseek():
    """Initialize the DeepSeek model using transformers"""
    global deepseek_model, deepseek_pipeline
    
    logger.info("DeepSeek model initialization skipped")
    return False

# Function to use DeepSeek model for content analysis
def analyze_with_deepseek(text, prompt_template=None):
    """
    Analyze text content using the DeepSeek model
    
    Args:
        text (str): The text to analyze
        prompt_template (str, optional): A template for the prompt. If None, a default template is used.
        
    Returns:
        float: A score between 0 and 1, or None if the analysis fails
    """
    logger.warning("DeepSeek model not initialized, returning default score")
    return 0.5  # Default score

# ----- DATA COLLECTION FUNCTIONS -----

def initialize_api_clients():
    """Initialize API clients for various data sources"""
    clients = {}
    
    # Twitter API setup
    try:
        twitter_auth = tweepy.OAuthHandler(
            os.getenv('q67mpjuZSDFTExV5Faq5G7aa6'),
            os.getenv('QJCjtJxn8ZMPEUJnxLiBnqb384gn5LurntsOennxyXzkajNmsr')
        )
        twitter_auth.set_access_token(
            os.getenv('1553087496006668288-fPFnLfrPKVFvL1IveaEM4hL7RIfMDb'),
            os.getenv('meIrqbysqZBOeCSETbnRUcVjaxYLPeFSVKAqB9uScIt8s')
        )
        clients['twitter'] = tweepy.API(twitter_auth)
        logger.info("Twitter API client initialized")
    except Exception as e:
        logger.warning(f"Twitter API client initialization failed: {e}")
        
    # Reddit API setup
    try:
        reddit = praw.Reddit(
            client_id=os.getenv('ykta_8jOq7qcOQTOz0r7fQ'),
            client_secret=os.getenv('SV4-rkT6xKGYBug7HoZu0LIfV0-ccg'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'InfluenceIQ Analytics v1.0')
        )
        clients['reddit'] = reddit
        logger.info("Reddit API client initialized")
    except Exception as e:
        logger.warning(f"Reddit API client initialization failed: {e}")
    
    # Add more API clients here (YouTube, Instagram, etc.)
    
    return clients

def collect_twitter_data(api, influencer_handle):
    """Collect data from Twitter for a given influencer"""
    try:
        # Get user profile data
        user = api.get_user(screen_name=influencer_handle)
        
        # Get recent tweets (limited to 200 by Twitter API)
        tweets = api.user_timeline(
            screen_name=influencer_handle, 
            count=200,
            tweet_mode='extended'
        )
        
        # Get mentions of the influencer
        mentions = api.search_tweets(
            q=f"@{influencer_handle}", 
            count=100,
            tweet_mode='extended'
        )
        
        # Process collected data
        tweet_data = [{
            'id': tweet.id,
            'text': tweet.full_text,
            'created_at': tweet.created_at.isoformat(),
            'likes': tweet.favorite_count,
            'retweets': tweet.retweet_count,
            'replies': 0  # Twitter API doesn't provide this directly
        } for tweet in tweets]
        
        mention_data = [{
            'id': mention.id,
            'text': mention.full_text,
            'created_at': mention.created_at.isoformat(),
            'user': mention.user.screen_name,
            'user_followers': mention.user.followers_count
        } for mention in mentions]
        
        return {
            'profile': {
                'handle': user.screen_name,
                'name': user.name,
                'followers': user.followers_count,
                'following': user.friends_count,
                'listed_count': user.listed_count,
                'verified': user.verified,
                'created_at': user.created_at.isoformat(),
                'description': user.description,
                'statuses_count': user.statuses_count
            },
            'tweets': tweet_data,
            'mentions': mention_data,
            'collected_at': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error collecting Twitter data for {influencer_handle}: {e}")
        return None

def collect_reddit_data(api, username):
    """Collect data from Reddit for a given influencer"""
    try:
        # Get user profile
        user = api.redditor(username)
        
        # Get recent submissions
        submissions = list(user.submissions.new(limit=100))
        
        # Get recent comments
        comments = list(user.comments.new(limit=100))
        
        # Process submissions data
        submission_data = [{
            'id': submission.id,
            'title': submission.title,
            'text': submission.selftext,
            'created_at': datetime.fromtimestamp(submission.created_utc).isoformat(),
            'score': submission.score,
            'upvote_ratio': submission.upvote_ratio,
            'num_comments': submission.num_comments,
            'subreddit': submission.subreddit.display_name
        } for submission in submissions]
        
        # Process comments data
        comment_data = [{
            'id': comment.id,
            'text': comment.body,
            'created_at': datetime.fromtimestamp(comment.created_utc).isoformat(),
            'score': comment.score,
            'subreddit': comment.subreddit.display_name
        } for comment in comments]
        
        return {
            'profile': {
                'username': username,
                'karma': user.link_karma + user.comment_karma,
                'created_at': datetime.fromtimestamp(user.created_utc).isoformat()
            },
            'submissions': submission_data,
            'comments': comment_data,
            'collected_at': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error collecting Reddit data for {username}: {e}")
        return None
def collect_news_mentions(entity_name, days=30):
    """Collect news mentions for an influencer using NewsAPI"""
    try:
        # Get API key from environment variables
        api_key = os.getenv('c871900c34bb4b7bbd869c76c2b7207c')
        
        if not api_key:
            logger.error("NewsAPI key not found in environment variables")
            return None
            
        url = 'https://newsapi.org/v2/everything'
        
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Make API request
        response = requests.get(url, params={
            'q': entity_name,
            'from': start_date,
            'to': end_date,
            'sortBy': 'relevancy',
            'language': 'en',
            'apiKey': api_key
        })
        
        if response.status_code != 200:
            logger.error(f"News API error: {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        
        # Process articles
        articles = [{
            'title': article['title'],
            'source': article['source']['name'],
            'url': article['url'],
            'published_at': article['publishedAt'],
            'description': article['description']
        } for article in data.get('articles', [])]
        
        return {
            'entity_name': entity_name,
            'total_results': data.get('totalResults', 0),
            'articles': articles,
            'collected_at': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error collecting news mentions for {entity_name}: {e}")
        return None

def store_data_in_database(data, collection_name, db_connector=None):
    """
    Store collected data in database
    This is a placeholder function - implement with your preferred database
    """
    # This would typically connect to MongoDB, PostgreSQL, etc.
    logger.info(f"Storing data in {collection_name} collection")
    
    # For demo purposes, we'll just save to a JSON file
    os.makedirs('data', exist_ok=True)
    filename = f"data/{collection_name}_{int(time.time())}.json"
    
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    logger.info(f"Data saved to {filename}")
    return filename

# ----- AI ANALYSIS FUNCTIONS -----

def extract_text_features(text_data):
    """Extract features from text content"""
    # Combine all text into a single string
    all_text = ' '.join(text_data)
    
    # Basic metrics
    word_count = len(all_text.split())
    avg_word_length = sum(len(word) for word in all_text.split()) / max(1, word_count)
    
    # Sentiment analysis
    sentiment = sentiment_analyzer.polarity_scores(all_text)
    
    return {
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'sentiment_positive': sentiment['pos'],
        'sentiment_negative': sentiment['neg'],
        'sentiment_neutral': sentiment['neu'],
        'sentiment_compound': sentiment['compound']
    }

def analyze_engagement_quality(engagement_data, text_key='text'):
    """Analyze the quality of engagement"""
    if not engagement_data:
        return {
            'sentiment_avg': 0,
            'diversity': 0,
            'depth': 0
        }
    
    # Extract text content
    texts = [item[text_key] for item in engagement_data if text_key in item]
    
    if not texts:
        logger.warning("No text content found in engagement data")
        return {
            'sentiment_avg': 0,
            'diversity': 0,
            'depth': 0
        }
    
    # Calculate sentiment scores
    sentiments = [sentiment_analyzer.polarity_scores(text)['compound'] for text in texts]
    sentiment_avg = sum(sentiments) / len(sentiments)
    
    # Use TF-IDF to analyze text diversity
    try:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        # Use fewer clusters if we have limited data
        n_clusters = min(5, len(texts) // 10) if len(texts) > 10 else 1
        
        if n_clusters > 1:
            # Cluster comments to identify diversity
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
            
            # Calculate cluster distribution as a diversity measure
            _, counts = np.unique(clusters, return_counts=True)
            diversity = 1 - (np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0
        else:
            diversity = 0
    except Exception as e:
        logger.error(f"Error in text clustering: {e}")
        diversity = 0
    
    # Calculate engagement depth (based on length, can be expanded)
    avg_length = sum(len(text.split()) for text in texts) / len(texts)
    depth = min(1.0, avg_length / 100)  # Normalize to 0-1 scale
    
    return {
        'sentiment_avg': sentiment_avg,
        'diversity': diversity,
        'depth': depth,
        'sample_size': len(texts)
    }

def calculate_credibility_score(influencer_data):
    """Calculate credibility score based on multiple factors"""
    # Extract relevant data
    profile = influencer_data.get('profile', {})
    
    # Basic profile credibility signals
    base_score = 0
    
    # Account verification adds credibility
    if profile.get('verified', False):
        base_score += 2
    
    # Account age contributes to credibility
    try:
        created_date = datetime.fromisoformat(profile.get('created_at', '2023-01-01'))
        account_age_years = (datetime.now() - created_date).days / 365
        age_score = min(3, account_age_years / 2)  # Max 3 points for 6+ years
    except:
        age_score = 0
    
    # Follower count contributes to credibility
    followers = profile.get('followers', 0)
    if followers > 1000000:  # 1M+
        follower_score = 2
    elif followers > 100000:  # 100K+
        follower_score = 1.5
    elif followers > 10000:  # 10K+
        follower_score = 1
    elif followers > 1000:  # 1K+
        follower_score = 0.5
    else:
        follower_score = 0.2
    
    # Calculate content consistency
    # This would be more sophisticated in a full implementation
    content_consistency = 0.5  # Placeholder
    
    # External validation (mentions, citations)
    mentions = influencer_data.get('mentions', [])
    mention_score = min(1.5, len(mentions) / 100)
    
    # Use DeepSeek AI for advanced content analysis if available
    content_quality = 0
    if deepseek_pipeline and 'tweets' in influencer_data:
        try:
            # Sample some content for analysis
            sample_texts = [t['text'] for t in influencer_data['tweets'][:5]]
            sample_text = "\n".join(sample_texts)
            
            # Analyze with DeepSeek
            content_quality = analyze_with_deepseek(sample_text)
            if content_quality is None:
                content_quality = 0.5  # Default if analysis fails
                
        except Exception as e:
            logger.error(f"DeepSeek analysis error: {e}")
            content_quality = 0.5
    
    # Combine all factors with appropriate weights
    raw_score = (
        base_score * 0.1 +
        age_score * 0.15 +
        follower_score * 0.15 +
        content_consistency * 0.2 +
        mention_score * 0.2 +
        content_quality * 0.2
    )
    
    # Scale to 0-10 range
    final_score = min(10, raw_score * 2)
    
    return final_score

def calculate_longevity_score(time_series_data):
    """Calculate longevity score based on consistency over time"""
    if not time_series_data:
        return 5.0  # Default score
    
    try:
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(time_series_data)
        
        # Ensure we have a date column
        if 'created_at' in df.columns:
            df['date'] = pd.to_datetime(df['created_at'])
        else:
            # No temporal data available
            return 5.0
        
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate time span in days
        time_span = (df['date'].max() - df['date'].min()).days
        
        # No meaningful time span
        if time_span <= 1:
            return 5.0
        
        # Calculate activity consistency
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_activity = df.groupby('year_month').size()
        
        # Calculate coefficient of variation (lower is more consistent)
        cv = monthly_activity.std() / monthly_activity.mean() if monthly_activity.mean() > 0 else 1
        consistency = 1 / (1 + cv)  # Transform to 0-1 scale where higher is better
        
        # Calculate growth or decay trend
        if len(monthly_activity) > 1:
            # Simple linear regression for trend
            x = np.arange(len(monthly_activity))
            y = monthly_activity.values
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize slope
            trend = 0.5 + 0.5 * (slope / monthly_activity.mean()) if monthly_activity.mean() > 0 else 0.5
            trend = max(0, min(1, trend))  # Bound to 0-1
        else:
            trend = 0.5
        
        # Calculate overall longevity score
        # Weight factors: time_span (30%), consistency (40%), trend (30%)
        time_factor = min(1, time_span / 365)  # Max out at 1 year
        
        raw_score = (
            time_factor * 0.3 +
            consistency * 0.4 +
            trend * 0.3
        )
        
        # Scale to 0-10
        longevity_score = raw_score * 10
        
        return longevity_score
    
    except Exception as e:
        logger.error(f"Error calculating longevity score: {e}")
        return 5.0  # Default score on error

def detect_suspicious_activity(user_activity):
    """Detect suspicious patterns in user activity"""
    if not user_activity or len(user_activity) < 10:
        return False
    
    try:
        # Extract features for anomaly detection
        features = []
        
        for activity in user_activity:
            # Example features (customize based on available data)
            feature_vector = [
                activity.get('time_of_day', 0),  # Hour (0-24)
                activity.get('day_of_week', 0),  # Day (0-6)
                activity.get('text_length', 0),  # Length of content
                activity.get('response_time', 0)  # Time to respond
            ]
            features.append(feature_vector)
        
        # Run anomaly detection
        if len(features) >= 10:  # Need enough samples
            detector = IsolationForest(contamination=0.1, random_state=42)
            predictions = detector.fit_predict(features)
            
            # Count anomalies (predicted as -1)
            anomaly_count = sum(1 for pred in predictions if pred == -1)
            anomaly_percentage = anomaly_count / len(predictions)
            
            # If more than 20% are anomalies, flag as suspicious
            return anomaly_percentage > 0.2
        
        return False
    
    except Exception as e:
        logger.error(f"Error in suspicious activity detection: {e}")
        return False

def calculate_overall_influence_score(influencer_data, field=None):
    """Calculate the overall influence score"""
    # Get component scores
    credibility = calculate_credibility_score(influencer_data)
    
    # Get time series data for longevity calculation
    time_series = []
    if 'tweets' in influencer_data:
        time_series = influencer_data['tweets']
    elif 'submissions' in influencer_data:
        time_series = influencer_data['submissions']
    
    longevity = calculate_longevity_score(time_series)
    
    # Get engagement data
    engagement_data = []
    if 'mentions' in influencer_data:
        engagement_data = influencer_data['mentions']
    
    engagement_quality = analyze_engagement_quality(engagement_data)
    engagement_score = engagement_quality['sentiment_avg'] * 0.3 + \
                       engagement_quality['diversity'] * 0.3 + \
                       engagement_quality['depth'] * 0.4
    engagement_score *= 10  # Scale to 0-10
    
    # Get field-specific weights
    weights = get_field_weights(field)
    
    # Calculate final score
    final_score = (
        weights['credibility'] * credibility +
        weights['longevity'] * longevity +
        weights['engagement'] * engagement_score
    )
    
    return {
        'overall_score': final_score,
        'components': {
            'credibility': credibility,
            'longevity': longevity,
            'engagement': engagement_score
        }
    }

def get_field_weights(field):
    """Get weighting factors based on the influencer's field"""
    weights = {
        'tech_entrepreneur': {
            'credibility': 0.40,
            'longevity': 0.35,
            'engagement': 0.25
        },
        'content_creator': {
            'credibility': 0.30,
            'longevity': 0.30,
            'engagement': 0.40
        },
        'athlete': {
            'credibility': 0.35,
            'longevity': 0.40,
            'engagement': 0.25
        },
        'musician': {
            'credibility': 0.30,
            'longevity': 0.45,
            'engagement': 0.25
        },
        'academic': {
            'credibility': 0.50,
            'longevity': 0.30,
            'engagement': 0.20
        }
    }
    
    # Return default weights if field not specified or unknown
    return weights.get(field, {
        'credibility': 0.35,
        'longevity': 0.35,
        'engagement': 0.30
    })

# ----- API ENDPOINTS -----

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/influencer/analyze', methods=['POST'])
def analyze_influencer():
    """
    Analyze an influencer based on handle/username
    Expected JSON payload:
    {
        "platform": "twitter",
        "handle": "username",
        "field": "tech_entrepreneur"  # Optional
    }
    """
    data = request.json
    
    # Validate request data
    if not data or 'platform' not in data or 'handle' not in data:
        return jsonify({
            'error': 'Missing required fields'
        }), 400
    
    platform = data['platform'].lower()
    handle = data['handle']
    field = data.get('field')
    
    # Check cache first
    cache_key = f"influencer:{platform}:{handle}"
    
    if redis_client:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            try:
                cached_result = json.loads(cached_data)
                cached_time = datetime.fromisoformat(cached_result.get('timestamp', '2000-01-01'))
                
                # If cache is less than 1 day old, return it
                if (datetime.now() - cached_time).days < 1:
                    logger.info(f"Returning cached result for {platform}:{handle}")
                    return jsonify(cached_result)
            except:
                logger.warning("Failed to parse cached data")
    
    # Initialize API clients
    api_clients = initialize_api_clients()
    
    # Collect data based on platform
    influencer_data = None
    
    if platform == 'twitter' and 'twitter' in api_clients:
        influencer_data = collect_twitter_data(api_clients['twitter'], handle)
    elif platform == 'reddit' and 'reddit' in api_clients:
        influencer_data = collect_reddit_data(api_clients['reddit'], handle)
    # Add more platforms here
    
    if not influencer_data:
        return jsonify({
            'error': f'Failed to collect data for {handle} on {platform}'
        }), 500
    
    # Store data (in a real implementation, this would go to a database)
    store_data_in_database(influencer_data, f"{platform}_data")
    
    # Calculate influence score
    score_data = calculate_overall_influence_score(influencer_data, field)
    
    # Prepare response
    result = {
        'platform': platform,
        'handle': handle,
        'field': field,
        'overall_score': score_data['overall_score'],
        'component_scores': score_data['components'],
        'timestamp': datetime.now().isoformat()
    }
    
    # Cache the result
    if redis_client:
        try:
            redis_client.setex(
                cache_key,
                86400,  # 24 hours in seconds
                json.dumps(result)
            )
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
    
    return jsonify(result)

@app.route('/api/influencers/top', methods=['GET'])
def get_top_influencers():
    """Get top influencers, with optional field filter"""
    field = request.args.get('field')
    limit = min(int(request.args.get('limit', 10)), 50)  # Max 50
    
    # In a real implementation, this would query a database
    # For demonstration, return some mock data
    mock_top_influencers = [
        {
            'platform': 'twitter',
            'handle': 'techleader',
            'name': 'Tech Leader',
            'field': 'tech_entrepreneur',
            'overall_score': 9.2,
            'component_scores': {
                'credibility': 9.5,
                'longevity': 8.7,
                'engagement': 9.3
            }
        },
        {
            'platform': 'twitter',
            'handle': 'contentmaster',
            'name': 'Content Master',
            'field': 'content_creator',
            'overall_score': 8.8,
            'component_scores': {
                'credibility': 8.2,
                'longevity': 8.5,
                'engagement': 9.8
            }
        },
        {
            'platform': 'twitter',
            'handle': 'sportsstar',
            'name': 'Sports Star',
            'field': 'athlete',
            'overall_score': 9.4,
            'component_scores': {
                'credibility': 9.7,
                'longevity': 9.6,
                'engagement': 8.9
            }
        }
    ]
    
    # Filter by field if specified
    if field:
        filtered_influencers = [i for i in mock_top_influencers if i['field'] == field]
    else:
        filtered_influencers = mock_top_influencers
    
    # Limit results
    results = filtered_influencers[:limit]
    
    return jsonify({
        'top_influencers': results,
        'count': len(results),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback on an influencer score"""
    data = request.json
    
    # Validate request data
    required_fields = ['platform', 'handle', 'user_id', 'score_opinion']
    if not data or not all(field in data for field in required_fields):
        return jsonify({
            'error': 'Missing required fields'
        }), 400
    
    # Store feedback (in a real implementation, this would go to a database)
    feedback_data = {
        'platform': data['platform'],
        'handle': data['handle'],
        'user_id': data['user_id'],
        'score_opinion': data['score_opinion'],
        'comments': data.get('comments', ''),
        'timestamp': datetime.now().isoformat()
    }
    
    store_data_in_database(feedback_data, 'user_feedback')
    
    return jsonify({
        'status': 'success',
        'message': 'Feedback submitted successfully',
        'timestamp': datetime.now().isoformat()
    })

# ----- SCHEDULING FUNCTIONS -----

def schedule_data_collection():
    """Set up scheduled tasks for data collection"""
    scheduler = BackgroundScheduler()
    
    # Schedule different collection tasks
    # In a real implementation, these would update a database
    
    # Example of scheduling Twitter data collection for top influencers
    def collect_top_twitter_data():
        api_clients = initialize_api_clients()
        if 'twitter' in api_clients:
            top_handles = ['elonmusk', 'BillGates', 'BarackObama']  
            for handle in top_handles:
                try:
                    data = collect_twitter_data(api_clients['twitter'], handle)
                    if data:
                        store_data_in_database(data, 'twitter_data')
                        logger.info(f"Collected scheduled data for Twitter:{handle}")
                except Exception as e:
                    logger.error(f"Scheduled collection failed for {handle}: {e}")
    
# Schedule tasks with different frequencies
    scheduler.add_job(collect_top_twitter_data, 'interval', hours=6)
    
    # Add more scheduled tasks as needed
    # e.g., Reddit data collection, news mentions, etc.
    
    scheduler.start()
    logger.info("Scheduled data collection tasks started")

# ----- MODEL TRAINING FUNCTIONS -----

def train_prediction_model(historical_data, target_column='engagement_rate'):
    """Train a model to predict influence factors"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Feature engineering
        # Add your custom feature engineering logic here
        
        # Prepare features and target
        X = df.drop([target_column, 'id', 'created_at'], axis=1, errors='ignore')
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        test_score = model.score(X_test, y_test)
        logger.info(f"Model trained with R² score: {test_score:.3f}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error training prediction model: {e}")
        return None

def save_model(model, filename='models/influence_model.pkl'):
    """Save a trained model to disk"""
    import pickle
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {filename}")

def load_model(filename='models/influence_model.pkl'):
    """Load a trained model from disk"""
    import pickle
    
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filename}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# ----- MAIN APPLICATION -----

def initialize_app():
    """Initialize the application"""
    logger.info("Initializing InfluenceIQ application")
    
    # Initialize DeepSeek AI
    initialize_deepseek()
    
    # Schedule data collection tasks
    schedule_data_collection()
    
    # Other initialization tasks can go here
    # ...
    
    logger.info("Initialization complete")

if __name__ == "__main__":
    initialize_app()
    
    # Start Flask app
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('DEBUG', 'False').lower() == 'true')

