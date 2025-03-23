import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import tweepy
import praw
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import redis
import pickle
import re
from g4f.client import Client

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("influenceiq.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Redis client with proper error handling
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD', None),
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Redis connection established")
except redis.ConnectionError:
    logger.warning("Redis server not available. Caching will be disabled.")
    redis_client = None
except Exception as e:
    logger.warning(f"Redis error: {e}. Caching will be disabled.")
    redis_client = None

# Initialize NLTK resources
nltk.download('vader_lexicon', quiet=True)
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize G4F client
g4f_client = None

def initialize_g4f_client():
    global g4f_client
    
    try:
        logger.info("Initializing G4F client...")
        g4f_client = Client()
        logger.info("G4F client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize G4F client: {e}")
        return False

def analyze_with_g4f(text, prompt_template=None):
    if not g4f_client:
        logger.warning("G4F client not initialized")
        return None
    
    try:
        if not prompt_template:
            prompt_template = "Analyze the following content and rate its quality on a scale of 0-1 based on expertise, depth, and informativeness:\n\n{content}\n\nProvide just a number between 0 and 1."
        
        prompt = prompt_template.format(content=text)
        
        response = g4f_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            web_search=False
        )
        
        response_text = response.choices[0].message.content
        
        # Parse the response to extract the score
        numeric_match = re.search(r"(\d+\.\d+|\d+)", response_text)
        if numeric_match:
            score = float(numeric_match.group(1))
            return min(1, max(0, score))
        
        logger.warning(f"Could not extract a valid score from G4F response: {response_text}")
        return 0.5
        
    except Exception as e:
        logger.error(f"Error analyzing with G4F: {e}")
        return None

def initialize_api_clients():
    clients = {}
    
    # Initialize Twitter API (using API v1.1)
    try:
        twitter_auth = tweepy.OAuthHandler(
            os.getenv('TWITTER_API_KEY', 'q67mpjuZSDFTExV5Faq5G7aa6'),
            os.getenv('TWITTER_API_SECRET', 'QJCjtJxn8ZMPEUJnxLiBnqb384gn5LurntsOennxyXzkajNmsr')
        )
        twitter_auth.set_access_token(
            os.getenv('TWITTER_ACCESS_TOKEN', '1553087496006668288-hKLTaKcr0Igj64VMMDvF3zN1hFOjon'),
            os.getenv('TWITTER_ACCESS_SECRET', 'ZiZ0Issa91bx0iZZSsTMuy7QYvbDNjuCrE6rbC72XYJ3K')
        )
        clients['twitter'] = tweepy.API(twitter_auth)
        logger.info("Twitter API client initialized")
    except Exception as e:
        logger.warning(f"Twitter API client initialization failed: {e}")
        
    try:
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID', 'ykta_8jOq7qcOQTOz0r7fQ'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET', 'SV4-rkT6xKGYBug7HoZu0LIfV0-ccg'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'InfluenceIQ Analytics v1.0')
        )
        clients['reddit'] = reddit
        logger.info("Reddit API client initialized")
    except Exception as e:
        logger.warning(f"Reddit API client initialization failed: {e}")
    
    return clients

def collect_twitter_data(api, influencer_handle):
    try:
        # Get user info using Twitter API v2
        user = api.get_user(screen_name=influencer_handle, tweet_mode='extended')
        
        # Get user's tweets (use correct Twitter API v2 method)
        tweets = api.user_timeline(
            screen_name=influencer_handle,
            count=100,
            tweet_mode='extended',
            include_rts=False
        )
        
        # Get mentions
        mentions = api.search_tweets(
            q=f"@{influencer_handle}", 
            count=100,
            tweet_mode='extended'
        )
        
        tweet_data = [{
            'id': tweet.id,
            'text': tweet.full_text,
            'created_at': tweet.created_at.isoformat(),
            'likes': tweet.favorite_count,
            'retweets': tweet.retweet_count,
            'replies': getattr(tweet, 'reply_count', 0)
        } for tweet in tweets]
        
        mention_data = [{
            'id': mention.id,
            'text': mention.full_text,
            'created_at': mention.created_at.isoformat(),
            'user': mention.user.screen_name,
            'user_followers': mention.user.followers_count
        } for mention in mentions]
        
        profile_image_url = user.profile_image_url_https.replace('_normal', '')
        
        return {
            'profile': {
                'handle': user.screen_name,
                'name': user.name,
                'followers': user.followers_count,
                'following': user.friends_count,
                'listed_count': user.listed_count,
                'verified': getattr(user, 'verified', False),
                'created_at': user.created_at.isoformat(),
                'description': user.description,
                'statuses_count': user.statuses_count,
                'profile_image_url': profile_image_url
            },
            'tweets': tweet_data,
            'mentions': mention_data,
            'collected_at': datetime.now().isoformat()
        }
    except tweepy.TweepyException as e:
        logger.error(f"Error collecting Twitter data for {influencer_handle}: {e}")
        return None
    except Exception as e:
        logger.error(f"General error collecting Twitter data for {influencer_handle}: {e}")
        return None

def collect_reddit_data(api, username):
    try:
        user = api.redditor(username)
        
        # Check if user exists
        try:
            _ = user.id
        except:
            logger.error(f"Reddit user {username} not found")
            return None
        
        submissions = list(user.submissions.new(limit=100))
        comments = list(user.comments.new(limit=100))
        
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
        
        comment_data = [{
            'id': comment.id,
            'text': comment.body,
            'created_at': datetime.fromtimestamp(comment.created_utc).isoformat(),
            'score': comment.score,
            'subreddit': comment.subreddit.display_name
        } for comment in comments]
        
        # Fixed: Safely get icon_img attribute
        icon_img = getattr(user, 'icon_img', '') if hasattr(user, 'icon_img') else "https://www.redditstatic.com/avatars/defaults/v2/avatar_default_1.png"
        
        return {
            'profile': {
                'username': username,
                'karma': user.link_karma + user.comment_karma,
                'created_at': datetime.fromtimestamp(user.created_utc).isoformat(),
                'profile_image_url': icon_img
            },
            'submissions': submission_data,
            'comments': comment_data,
            'collected_at': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error collecting Reddit data for {username}: {e}")
        return None

def collect_news_mentions(entity_name, days=30):
    try:
        # Fixed: Use proper environment variable
        api_key = os.getenv('NEWS_API_KEY', 'c871900c34bb4b7bbd869c76c2b7207c')
        
        if not api_key:
            logger.error("NewsAPI key not found in environment variables")
            return None
            
        url = 'https://newsapi.org/v2/everything'
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        
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
        
        articles = [{
            'title': article.get('title', ''),
            'source': article.get('source', {}).get('name', ''),
            'url': article.get('url', ''),
            'published_at': article.get('publishedAt', ''),
            'description': article.get('description', '')
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

def store_data_in_database(data, collection_name):
    logger.info(f"Storing data in {collection_name} collection")
    
    os.makedirs('data', exist_ok=True)
    filename = f"data/{collection_name}_{int(time.time())}.json"
    
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    logger.info(f"Data saved to {filename}")
    return filename

def extract_text_features(text_data):
    if not text_data:
        return {
            'word_count': 0,
            'avg_word_length': 0,
            'sentiment_positive': 0,
            'sentiment_negative': 0,
            'sentiment_neutral': 0,
            'sentiment_compound': 0
        }
        
    all_text = ' '.join(text_data)
    
    words = all_text.split()
    word_count = len(words)
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    
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
    if not engagement_data:
        return {
            'sentiment_avg': 0,
            'diversity': 0,
            'depth': 0,
            'sample_size': 0
        }
    
    texts = [item.get(text_key, '') for item in engagement_data if text_key in item and item.get(text_key)]
    
    if not texts:
        logger.warning("No text content found in engagement data")
        return {
            'sentiment_avg': 0,
            'diversity': 0,
            'depth': 0,
            'sample_size': 0
        }
    
    sentiments = [sentiment_analyzer.polarity_scores(text)['compound'] for text in texts]
    sentiment_avg = sum(sentiments) / len(sentiments)
    
    try:
        if len(texts) >= 5:  # Need minimum texts for meaningful TF-IDF
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            X = vectorizer.fit_transform(texts)
            
            n_clusters = min(5, len(texts) // 10) if len(texts) > 10 else 1
            
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(X)
                
                _, counts = np.unique(clusters, return_counts=True)
                diversity = 1 - (np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0
            else:
                diversity = 0
        else:
            diversity = 0
    except Exception as e:
        logger.error(f"Error in text clustering: {e}")
        diversity = 0
    
    word_counts = [len(text.split()) for text in texts]
    avg_length = sum(word_counts) / max(1, len(texts))
    depth = min(1.0, avg_length / 100)
    
    return {
        'sentiment_avg': sentiment_avg,
        'diversity': diversity,
        'depth': depth,
        'sample_size': len(texts)
    }

def calculate_credibility_score(influencer_data):
    if not influencer_data:
        return 5.0
        
    profile = influencer_data.get('profile', {})
    
    base_score = 0
    
    if profile.get('verified', False):
        base_score += 2
    
    try:
        created_date = datetime.fromisoformat(profile.get('created_at', '2023-01-01'))
        account_age_years = (datetime.now() - created_date).days / 365
        age_score = min(3, account_age_years / 2)
    except Exception as e:
        logger.error(f"Error calculating account age: {e}")
        age_score = 0
    
    followers = profile.get('followers', 0)
    if followers > 1000000:
        follower_score = 2
    elif followers > 100000:
        follower_score = 1.5
    elif followers > 10000:
        follower_score = 1
    elif followers > 1000:
        follower_score = 0.5
    else:
        follower_score = 0.2
    
    content_consistency = 0.5
    
    mentions = influencer_data.get('mentions', [])
    mention_score = min(1.5, len(mentions) / 100)
    
    content_quality = 0
    if g4f_client:
        try:
            sample_text = ""
            if 'tweets' in influencer_data and influencer_data['tweets']:
                sample_texts = [t.get('text', '') for t in influencer_data['tweets'][:5] if 'text' in t]
                sample_text = "\n".join(sample_texts)
            elif 'submissions' in influencer_data and influencer_data['submissions']:
                sample_texts = [s.get('text', '') for s in influencer_data['submissions'][:5] if 'text' in s]
                sample_text = "\n".join(sample_texts)
            
            if not sample_text:
                sample_text = "No content available for analysis."
            
            quality_result = analyze_with_g4f(sample_text)
            content_quality = quality_result if quality_result is not None else 0.5
                
        except Exception as e:
            logger.error(f"G4F analysis error: {e}")
            content_quality = 0.5
    
    raw_score = (
        base_score * 0.1 +
        age_score * 0.15 +
        follower_score * 0.15 +
        content_consistency * 0.2 +
        mention_score * 0.2 +
        content_quality * 0.2
    )
    
    final_score = min(10, raw_score * 2)
    
    return final_score

def calculate_longevity_score(time_series_data):
    if not time_series_data:
        return 5.0
    
    try:
        df = pd.DataFrame(time_series_data)
        
        if 'created_at' in df.columns:
            df['date'] = pd.to_datetime(df['created_at'], errors='coerce')
            # Drop rows with invalid dates
            df = df.dropna(subset=['date'])
        else:
            return 5.0
            
        if df.empty:
            return 5.0
        
        df = df.sort_values('date')
        
        time_span = (df['date'].max() - df['date'].min()).days
        
        if time_span <= 1:
            return 5.0
        
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_activity = df.groupby('year_month').size()
        
        if monthly_activity.empty:
            return 5.0
            
        if monthly_activity.mean() > 0:
            cv = monthly_activity.std() / monthly_activity.mean()
            consistency = 1 / (1 + cv)
        else:
            consistency = 0
        
        if len(monthly_activity) > 1:
            x = np.arange(len(monthly_activity))
            y = monthly_activity.values
            slope = np.polyfit(x, y, 1)[0]
            
            if monthly_activity.mean() > 0:
                trend = 0.5 + 0.5 * (slope / monthly_activity.mean())
                trend = max(0, min(1, trend))
            else:
                trend = 0.5
        else:
            trend = 0.5
        
        time_factor = min(1, time_span / 365)
        
        raw_score = (
            time_factor * 0.3 +
            consistency * 0.4 +
            trend * 0.3
        )
        
        longevity_score = raw_score * 10
        
        return max(0, min(10, longevity_score))
    
    except Exception as e:
        logger.error(f"Error calculating longevity score: {e}")
        return 5.0

def detect_suspicious_activity(user_activity):
    if not user_activity or len(user_activity) < 10:
        return False
    
    try:
        features = []
        
        for activity in user_activity:
            # Check if all required keys exist
            if not all(k in activity for k in ['time_of_day', 'day_of_week', 'text_length', 'response_time']):
                continue
                
            feature_vector = [
                activity.get('time_of_day', 0),
                activity.get('day_of_week', 0),
                activity.get('text_length', 0),
                activity.get('response_time', 0)
            ]
            features.append(feature_vector)
        
        if len(features) >= 10:
            detector = IsolationForest(contamination=0.1, random_state=42)
            predictions = detector.fit_predict(features)
            
            anomaly_count = sum(1 for pred in predictions if pred == -1)
            anomaly_percentage = anomaly_count / len(predictions)
            
            return anomaly_percentage > 0.2
        
        return False
    
    except Exception as e:
        logger.error(f"Error in suspicious activity detection: {e}")
        return False

def get_field_weights(field):
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
    
    return weights.get(field, {
        'credibility': 0.35,
        'longevity': 0.35,
        'engagement': 0.30
    })

def calculate_overall_influence_score(influencer_data, field=None):
    if not influencer_data:
        return {
            'overall_score': 5.0,
            'components': {
                'credibility': 5.0,
                'longevity': 5.0,
                'engagement': 5.0
            }
        }
        
    credibility = calculate_credibility_score(influencer_data)
    
    time_series = []
    if 'tweets' in influencer_data:
        time_series = influencer_data['tweets']
    elif 'submissions' in influencer_data:
        time_series = influencer_data['submissions']
    
    longevity = calculate_longevity_score(time_series)
    
    engagement_data = []
    if 'mentions' in influencer_data:
        engagement_data = influencer_data['mentions']
    
    engagement_quality = analyze_engagement_quality(engagement_data)
    engagement_score = engagement_quality['sentiment_avg'] * 0.3 + \
                       engagement_quality['diversity'] * 0.3 + \
                       engagement_quality['depth'] * 0.4
    engagement_score *= 10
    engagement_score = max(0, min(10, engagement_score))
    
    weights = get_field_weights(field)
    
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

def analyze_influencer(platform, handle, field=None):
    if not handle:
        logger.error("Empty handle provided")
        return None
        
    cache_key = f"influencer:{platform}:{handle}"
    
    if redis_client:
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                try:
                    cached_result = json.loads(cached_data)
                    cached_time = datetime.fromisoformat(cached_result.get('timestamp', '2000-01-01'))
                    
                    if (datetime.now() - cached_time).days < 1:
                        logger.info(f"Returning cached result for {platform}:{handle}")
                        return cached_result
                except Exception as e:
                    logger.warning(f"Failed to parse cached data: {e}")
        except Exception as e:
            logger.warning(f"Redis error while fetching cached data: {e}")
    
    api_clients = initialize_api_clients()
    
    influencer_data = None
    
    if platform == 'twitter' and 'twitter' in api_clients:
        influencer_data = collect_twitter_data(api_clients['twitter'], handle)
    elif platform == 'reddit' and 'reddit' in api_clients:
        influencer_data = collect_reddit_data(api_clients['reddit'], handle)
    
    if not influencer_data:
        return None
    
    store_data_in_database(influencer_data, f"{platform}_data")
    
    score_data = calculate_overall_influence_score(influencer_data, field)
    
    # Added profile_image_url to result
    profile_image_url = influencer_data.get('profile', {}).get('profile_image_url', '')
    
    result = {
        'platform': platform,
        'handle': handle,
        'name': influencer_data.get('profile', {}).get('name', handle),
        'field': field,
        'overall_score': score_data['overall_score'],
        'component_scores': score_data['components'],
        'profile_image_url': profile_image_url,
        'timestamp': datetime.now().isoformat()
    }
    
    if redis_client:
        try:
            redis_client.setex(
                cache_key,
                86400,
                json.dumps(result)
            )
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
    
    return result

def get_mock_top_influencers(field=None, limit=10):
    mock_top_influencers = [
        {
            'platform': 'twitter',
            'handle': 'techleader',
            'name': 'Jane Smith',
            'field': 'tech_entrepreneur',
            'overall_score': 9.0,
            'component_scores': {
                'credibility': 9.2,
                'longevity': 8.7,
                'engagement': 9.0
            },
            'profile_image_url': 'https://randomuser.me/api/portraits/women/44.jpg'
        },
        {
            'platform': 'twitter',
            'handle': 'contentmaster',
            'name': 'John Doe',
            'field': 'content_creator',
            'overall_score': 8.2,
            'component_scores': {
                'credibility': 8.5,
                'longevity': 7.2,
                'engagement': 8.7
            },
            'profile_image_url': 'https://randomuser.me/api/portraits/men/32.jpg'
        },
        {
            'platform': 'twitter',
            'handle': 'sportsstar',
            'name': 'Alex Johnson',
            'field': 'athlete',
            'overall_score': 9.4,
            'component_scores': {
                'credibility': 9.7,
                'longevity': 8.9,
                'engagement': 9.5
            },
            'profile_image_url': 'https://randomuser.me/api/portraits/men/22.jpg'
        }
    ]
    
    if field:
        filtered_influencers = [i for i in mock_top_influencers if i['field'] == field]
    else:
        filtered_influencers = mock_top_influencers
    
    results = filtered_influencers[:limit]
    
    return results

def display_influencer_card(influencer):
    if not influencer:
        st.warning("No influencer data available")
        return
        
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        profile_img = influencer.get('profile_image_url', 'https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp')
        st.image(profile_img, width=100)
    
    with col2:
        st.markdown(f"### {influencer.get('name', 'Unknown')}")
        field = influencer.get('field', '')
        if field:
            st.markdown(f"*{field.replace('_', ' ').title()}*")
        
        # Component scores with icons
        component_scores = influencer.get('component_scores', {})
        st.markdown(f"🔵 Credibility: {component_scores.get('credibility', 0):.1f}/10")
        st.markdown(f"🟢 Longevity: {component_scores.get('longevity', 0):.1f}/10")
        st.markdown(f"🟠 Engagement: {component_scores.get('engagement', 0):.1f}/10")
    
    with col3:
        overall_score = influencer.get('overall_score', 0)
        st.markdown(f"## {overall_score:.1f}")
        st.markdown("*overall score*")

def initialize_app():
    logger.info("Initializing InfluenceIQ application")
    initialize_g4f_client()
    logger.info("Initialization complete")

def main():
    initialize_app()
    
    st.set_page_config(
        page_title="InfluenceIQ",
        page_icon="📊",
        layout="wide"
    )
    
    # Header with search bar
    col1, col2 = st.columns([3, 2])
    with col1:
        st.title("InfluenceIQ")
    with col2:
        search_query = st.text_input("", placeholder="Search for an influencer...")
    
    # Main content area with sidebar
    st.sidebar.header("Categories")
    categories = ["All Influencers", "Tech Entrepreneurs", "Content Creators", "Athletes", "Musicians", "Academics"]
    selected_category = st.sidebar.radio("", categories)
    
    st.sidebar.header("Filters")
    filter_options = ["Rising Stars", "Legacy Figures"]
    selected_filters = st.sidebar.multiselect("", filter_options)
    
    # Map selected category to field value
    field_mapping = {
        "Tech Entrepreneurs": "tech_entrepreneur",
        "Content Creators": "content_creator",
        "Athletes": "athlete",
        "Musicians": "musician",
        "Academics": "academic"
    }
    
    selected_field = field_mapping.get(selected_category, None)
    
    # Main content - Top Influencers
    st.header("Top Influencers")
    
    # Get influencer data
    influencers = get_mock_top_influencers(field=selected_field)
    
    # Display influencers in a grid
    if influencers:
        cols = st.columns(min(3, len(influencers)))
        for i, influencer in enumerate(influencers[:3]):
            with cols[i]:
                display_influencer_card(influencer)
    else:
        st.info("No influencers found")
    
    st.markdown("Click on any profile for detailed influence metrics")
    
# Handle search functionality
    if search_query:
            st.subheader(f"Search Results for '{search_query}'")
            # Default to Twitter search if platform not specified
            search_parts = search_query.split(':')
            if len(search_parts) > 1:
                platform = search_parts[0].lower()
                handle = search_parts[1].strip()
            else:
                platform = 'twitter'
                handle = search_query.strip()
            
            if platform in ['twitter', 'reddit']:
                st.info(f"Analyzing {platform} user: {handle}")
                
                with st.spinner(f"Analyzing {handle}'s influence..."):
                    result = analyze_influencer(platform, handle, field=selected_field)
                
                if result:
                    display_influencer_card(result)
                else:
                    st.error(f"Could not retrieve data for {handle} on {platform}")
            else:
                st.error("Supported platforms are Twitter and Reddit. Use format platform:handle")
    
    # Analysis section
    st.header("Analyze New Influencer")
    col1, col2 = st.columns(2)
    
    with col1:
        new_platform = st.selectbox("Platform", ["Twitter", "Reddit"])
    
    with col2:
        new_handle = st.text_input("Handle", placeholder="Enter username without @")
    
    new_field = st.selectbox("Field", [
        "Tech Entrepreneur", 
        "Content Creator", 
        "Athlete", 
        "Musician", 
        "Academic"
    ])
    
    if st.button("Analyze"):
        if new_handle:
            field_value = new_field.lower().replace(" ", "_")
            platform_value = new_platform.lower()
            
            with st.spinner(f"Analyzing {new_handle}'s influence..."):
                result = analyze_influencer(platform_value, new_handle, field=field_value)
            
            if result:
                st.success("Analysis complete!")
                display_influencer_card(result)
                
                # Display additional metrics
                st.subheader("Detailed Metrics")
                
                component_scores = result.get('component_scores', {})
                
                metrics_cols = st.columns(3)
                with metrics_cols[0]:
                    st.metric("Credibility", f"{component_scores.get('credibility', 0):.1f}/10")
                with metrics_cols[1]:
                    st.metric("Longevity", f"{component_scores.get('longevity', 0):.1f}/10")
                with metrics_cols[2]:
                    st.metric("Engagement", f"{component_scores.get('engagement', 0):.1f}/10")
                
                # Placeholder for historical data chart
                st.subheader("Influence Trend")
                st.line_chart({
                    "Influence Score": [5.0, 5.2, 5.8, 6.3, 7.1, result.get('overall_score', 0)]
                })
            else:
                st.error(f"Could not retrieve data for {new_handle} on {platform_value}")
        else:
            st.warning("Please enter a handle to analyze")
    
    # Footer
    st.markdown("---")
    st.markdown("InfluenceIQ - Analytics for identifying authentic online influence")

if __name__ == "__main__":
    main()