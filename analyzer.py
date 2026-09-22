import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
import emoji

# Default stopwords list tailored for chat analysis
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't", 'ok', 'okay', 'yeah', 'yes', 'yep',
    'hi', 'hello', 'hey', 'lol', 'omg', 'bro', 'haha', 'hahaha', 'k', 'kk', 'media', 'omitted',
    'message', 'deleted', 'this', 'that', 'also', 'like', 'get', 'got', 'good', 'well',
    'one', 'know', 'see', 'come', 'go', 'going', 'let', 'make', 'take', 'want', 'tell'
}

def filter_by_user(df: pd.DataFrame, selected_user: str) -> pd.DataFrame:
    """Filters dataframe by selected user or returns all non-group notification messages."""
    if selected_user != 'Overall':
        return df[df['user'] == selected_user]
    return df[df['user'] != 'group_notification']

def fetch_stats(selected_user: str, df: pd.DataFrame):
    """Calculates top-level KPI metrics."""
    filtered_df = filter_by_user(df, selected_user)

    num_messages = filtered_df.shape[0]
    
    # Calculate words excluding media placeholders
    text_df = filtered_df[~filtered_df['is_media']]
    words = []
    for msg in text_df['message']:
        words.extend(msg.split())
    total_words = len(words)

    # Number of media messages
    num_media_messages = filtered_df['is_media'].sum()

    # Number of links shared
    num_links = filtered_df['url_count'].sum()

    # Number of unique users (if overall)
    num_users = filtered_df['user'].nunique() if selected_user == 'Overall' else 1

    return {
        'total_messages': int(num_messages),
        'total_words': int(total_words),
        'media_count': int(num_media_messages),
        'links_count': int(num_links),
        'users_count': int(num_users)
    }

def most_busy_users(df: pd.DataFrame):
    """Returns top active users and their percentage of total group conversation."""
    clean_df = df[df['user'] != 'group_notification']
    user_counts = clean_df['user'].value_counts().head(10)
    user_percent = (clean_df['user'].value_counts(normalize=True) * 100).round(2).reset_index()
    user_percent.columns = ['user', 'percent']
    return user_counts, user_percent

def create_wordcloud(selected_user: str, df: pd.DataFrame):
    """Generates a WordCloud object for text visualization."""
    filtered_df = filter_by_user(df, selected_user)
    text_df = filtered_df[~filtered_df['is_media']]

    words = []
    for message in text_df['message']:
        for word in message.lower().split():
            clean_word = word.strip('.,!?:;"\'()[]{}')
            if clean_word and clean_word not in STOP_WORDS and not emoji.is_emoji(clean_word) and not clean_word.startswith('http'):
                words.append(clean_word)

    if not words:
        words = ['no_data']

    text = " ".join(words)
    wc = WordCloud(
        width=800,
        height=450,
        min_font_size=10,
        background_color='#0E1117',
        colormap='cool',
        contour_width=1,
        contour_color='#3b82f6'
    )
    return wc.generate(text)

def most_common_words(selected_user: str, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Returns the top N most frequent words in a clean DataFrame."""
    filtered_df = filter_by_user(df, selected_user)
    text_df = filtered_df[~filtered_df['is_media']]

    words = []
    for message in text_df['message']:
        for word in message.lower().split():
            clean_word = word.strip('.,!?:;"\'()[]{}')
            if clean_word and len(clean_word) > 2 and clean_word not in STOP_WORDS and not emoji.is_emoji(clean_word) and not clean_word.startswith('http'):
                words.append(clean_word)

    word_counts = Counter(words).most_common(top_n)
    return pd.DataFrame(word_counts, columns=['word', 'count'])

def emoji_helper(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Extracts and counts emojis used in the conversation."""
    filtered_df = filter_by_user(df, selected_user)

    emojis = []
    for emoji_list in filtered_df['emojis']:
        emojis.extend(emoji_list)

    if not emojis:
        return pd.DataFrame(columns=['emoji', 'count'])

    emoji_counts = Counter(emojis).most_common()
    emoji_df = pd.DataFrame(emoji_counts, columns=['emoji', 'count'])
    return emoji_df

def monthly_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Generates message activity aggregated by month-year."""
    filtered_df = filter_by_user(df, selected_user)

    timeline = filtered_df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline

def daily_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Generates daily message activity timeline."""
    filtered_df = filter_by_user(df, selected_user)
    daily_stats = filtered_df.groupby('only_date').count()['message'].reset_index()
    daily_stats.rename(columns={'only_date': 'date', 'message': 'message_count'}, inplace=True)
    return daily_stats

def week_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    """Returns message counts broken down by day of the week."""
    filtered_df = filter_by_user(df, selected_user)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    counts = filtered_df['day_name'].value_counts()
    return counts.reindex(days_order, fill_value=0)

def month_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    """Returns message counts broken down by month."""
    filtered_df = filter_by_user(df, selected_user)
    months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']
    counts = filtered_df['month'].value_counts()
    return counts.reindex(months_order, fill_value=0)

def activity_heatmap(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Generates a Day of Week vs Hourly Period activity matrix for heatmap visualization."""
    filtered_df = filter_by_user(df, selected_user)
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    periods = [f"{h:02d}-{(h+1):02d}" if h < 23 else "23-00" for h in range(24)]
    
    heatmap_df = filtered_df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    
    # Reindex for consistent structure
    heatmap_df = heatmap_df.reindex(index=days_order, fill_value=0)
    for p in periods:
        if p not in heatmap_df.columns:
            heatmap_df[p] = 0
    heatmap_df = heatmap_df[sorted(heatmap_df.columns)]
    return heatmap_df

# NLP Sentiment Analysis
def calculate_sentiment(text: str) -> float:
    """Returns sentiment polarity between -1.0 (very negative) and +1.0 (very positive)."""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except Exception:
        return 0.0

def analyze_sentiment(selected_user: str, df: pd.DataFrame):
    """
    Performs comprehensive sentiment analysis on the chat dataset.
    Returns categorized percentages, user-wise sentiment breakdown, and timeline.
    """
    filtered_df = filter_by_user(df, selected_user).copy()
    text_df = filtered_df[~filtered_df['is_media']].copy()

    if text_df.empty:
        return {
            'overall_score': 0.0,
            'pos_percent': 0.0,
            'neu_percent': 100.0,
            'neg_percent': 0.0,
            'user_sentiment': pd.DataFrame(),
            'sentiment_timeline': pd.DataFrame()
        }

    # Calculate polarity for each message
    text_df['polarity'] = text_df['message'].apply(calculate_sentiment)

    def classify_sentiment(score):
        if score > 0.05:
            return 'Positive'
        elif score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    text_df['sentiment_label'] = text_df['polarity'].apply(classify_sentiment)

    total_msgs = len(text_df)
    pos_count = (text_df['sentiment_label'] == 'Positive').sum()
    neu_count = (text_df['sentiment_label'] == 'Neutral').sum()
    neg_count = (text_df['sentiment_label'] == 'Negative').sum()

    pos_pct = round((pos_count / total_msgs) * 100, 2) if total_msgs > 0 else 0
    neu_pct = round((neu_count / total_msgs) * 100, 2) if total_msgs > 0 else 0
    neg_pct = round((neg_count / total_msgs) * 100, 2) if total_msgs > 0 else 0

    overall_score = round(float(text_df['polarity'].mean()), 3)

    # User-wise sentiment
    user_sentiment = text_df.groupby('user').agg(
        avg_polarity=('polarity', 'mean'),
        positive_msgs=('sentiment_label', lambda x: (x == 'Positive').sum()),
        neutral_msgs=('sentiment_label', lambda x: (x == 'Neutral').sum()),
        negative_msgs=('sentiment_label', lambda x: (x == 'Negative').sum()),
        total_msgs=('sentiment_label', 'count')
    ).reset_index()
    user_sentiment['avg_polarity'] = user_sentiment['avg_polarity'].round(3)

    # Sentiment over time
    timeline_sentiment = text_df.groupby('only_date').agg(
        avg_polarity=('polarity', 'mean'),
        message_count=('message', 'count')
    ).reset_index()
    timeline_sentiment.rename(columns={'only_date': 'date'}, inplace=True)

    return {
        'overall_score': overall_score,
        'pos_percent': pos_pct,
        'neu_percent': neu_pct,
        'neg_percent': neg_pct,
        'user_sentiment': user_sentiment,
        'sentiment_timeline': timeline_sentiment
    }
