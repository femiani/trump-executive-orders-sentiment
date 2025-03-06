import praw
import pandas as pd
from datetime import datetime, timedelta
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

### Scrape
# Set up Reddit API credentials using PRAW
reddit = praw.Reddit(
    user_agent="MyPrawApp", 
    client_id="bYbwhw5MbGOjGAJssWwjwQ", 
    client_secret="jXSFTAT2VVz4v6I6ZrYM2UIKrfOi-g", 
    username="samanalyze", 
    password="create yours"
)
# List of subreddits to scrape from
subreddits = ["politics", "conservative", "news"]

def search_posts(keyword, subreddits, after_date):
    posts = []
    # Convert after_date to Unix timestamp
    after_timestamp = int(datetime.strptime(after_date, "%Y-%m-%d").timestamp())
    
    for sub in subreddits:
        subreddit_obj = reddit.subreddit(sub)
        # PRAW's search method does not have a direct date filter, so we iterate and filter manually.
        for submission in subreddit_obj.search(keyword, sort='new', limit=500):
            if submission.created_utc >= after_timestamp:
                # Convert post timestamp to readable format
                post_datetime = datetime.fromtimestamp(submission.created_utc).strftime("%Y-%m-%d %H:%M:%S")
                # Load all comments for this submission (flatten the tree)
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                    # Convert comment timestamp to readable format
                    comment_datetime = datetime.fromtimestamp(comment.created_utc).strftime("%Y-%m-%d %H:%M:%S")
                    posts.append({
                        'post_id': submission.id,
                        'post_title': submission.title,
                        'post_upvote_ratio': submission.upvote_ratio,
                        'post_timestamp': submission.created_utc,
                        'num_comments': submission.num_comments,
                        'post_datetime': post_datetime,
                        'comment_id': comment.id,
                        'comment_body': comment.body,
                        'comment_timestamp': comment.created_utc,
                        'comment_datetime': comment_datetime,
                        'subreddit': submission.subreddit.display_name
                    })
    return posts

# -----------------------------
# Scrape posts about Trump's election win after 2025-01-20
# -----------------------------
election_posts = search_posts("Trump", subreddits, "2025-01-20")
print(f"Scraped {len(election_posts)} posts about Trump's election win.")

# -----------------------------
# Scrape posts about executive orders after 2025-01-20
# -----------------------------
executive_posts = search_posts("executive order", subreddits, "2025-01-20")
print(f"Scraped {len(executive_posts)} posts about executive orders.")

# -----------------------------
# Save the scraped data to CSV files
# -----------------------------
if election_posts:
    election_df = pd.DataFrame(election_posts)
    election_df.to_csv("election_posts.csv", index=False)
    print("Saved election posts to election_posts.csv")
else:
    print("No election posts scraped.")

if executive_posts:
    executive_df = pd.DataFrame(executive_posts)
    executive_df.to_csv("executive_posts.csv", index=False)
    print("Saved executive posts to executive_posts.csv")
else:
    print("No executive posts scraped.")

### Clean
def clean_text(text):
    """
    Extensively cleans a text string:
      - Expands contractions
      - Removes URLs
      - Removes non-alphabetic characters
      - Converts text to lowercase
      - Removes extra whitespace
      - Removes English stopwords
      - Lemmatizes words
    """
    if not isinstance(text, str):
        return ""
    
    # Fix encoding issues
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Expand contractions
    text = contractions.fix(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'/r/\w+', '', text)  # Remove subreddit links
    
    # Remove non-alphabet characters (keeping spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove Reddit-specific markup
    text = re.sub(r'\&gt\;|\&lt\;|\&amp\;', '', text)  # HTML entities
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Markdown links
    text = re.sub(r'\*{1,3}', '', text)  # Bold/italic markers
    
    # Convert to lowercase and remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(words)

# Load scraped data from CSV files
df_election = pd.read_csv("election_posts.csv")
df_executive = pd.read_csv("executive_posts.csv")

# Define a list of text columns to clean if they exist
text_columns = ['post_title', 'comment_body']

# Function to clean specified columns in a DataFrame and add new cleaned columns
def clean_dataframe(df):
    for col in text_columns:
        if col in df.columns:
            df[f'clean_{col}'] = df[col].apply(clean_text)
    # Drop only the columns that were cleaned
    df.drop(columns=[col for col in text_columns if col in df.columns], inplace=True)
    return df

# Clean both DataFrames
df_election = clean_dataframe(df_election)
df_executive = clean_dataframe(df_executive)

# Save the cleaned data to new CSV files
df_election.to_csv("election_posts_clean.csv", index=False)
df_executive.to_csv("executive_posts_clean.csv", index=False)

print("Cleaned data saved to 'election_posts_clean.csv' and 'executive_posts_clean.csv'")

### Analyze
sia = SentimentIntensityAnalyzer()

# Load cleaned data
df_election = pd.read_csv("election_posts_clean.csv")
df_executive = pd.read_csv("executive_posts_clean.csv")

# Add identifier for dataset type
df_election['dataset_type'] = 'election'
df_executive['dataset_type'] = 'executive'

# Combine the datasets
combined_df = pd.concat([df_election, df_executive], ignore_index=True)

# Convert 'post_datetime' (already in readable format) to datetime object
combined_df['post_datetime'] = pd.to_datetime(combined_df['post_datetime'])

# Function to calculate sentiment scores with handling for non-string input
def get_sentiment_scores(text):
    if not isinstance(text, str):
        text = ""
    scores = sia.polarity_scores(text)
    return pd.Series({
        'compound': scores['compound'],
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu']
    })
    
# Define the text columns to analyze sentiment from
text_columns = ['clean_post_title', 'clean_comment_body']

# Apply sentiment analysis for each text column and join the results
for col in text_columns:
    if col in combined_df.columns:
        sentiment_scores = combined_df[col].apply(get_sentiment_scores).add_prefix(f'{col}_')
        combined_df = combined_df.join(sentiment_scores)

# Compute overall sentiment as the average of all available compound scores
compound_cols = [col for col in combined_df.columns if col.endswith('compound')]
combined_df['overall_sentiment'] = combined_df[compound_cols].mean(axis=1)

# ----------------------------
# Executive Order Impact Analysis
# ----------------------------
# Update the executive orders dictionary with the correct names and dates
executive_orders = {
    'Federal Hiring Freeze': '2025-02-01',
    'Remote Work Ban': '2025-02-15',
    'Doge Department Lunch': '2025-03-01'
}

impact_results = []

for order_name, order_date_str in executive_orders.items():
    order_date = pd.to_datetime(order_date_str)
    # Filter posts within 3 days before and after the order date using 'post_datetime'
    mask = (combined_df['post_datetime'] >= order_date - timedelta(days=3)) & \
           (combined_df['post_datetime'] <= order_date + timedelta(days=3))
    period_data = combined_df.loc[mask].copy()
    
    if period_data.empty:
        pre_sentiment = np.nan
        post_sentiment = np.nan
        pct_change = np.nan
    else:
        # Assign period as 'pre' or 'post'
        period_data['period'] = np.where(period_data['post_datetime'] < order_date, 'pre', 'post')
        sentiment_group = period_data.groupby('period')['overall_sentiment'].mean()
        pre_sentiment = sentiment_group.get('pre', np.nan)
        post_sentiment = sentiment_group.get('post', np.nan)
        if pd.isna(pre_sentiment) or pd.isna(post_sentiment) or pre_sentiment == 0:
            pct_change = np.nan
        else:
            pct_change = (post_sentiment - pre_sentiment) / pre_sentiment * 100
    
    impact_results.append({
        'executive_order': order_name,
        'order_date': order_date,
        'pre_sentiment': pre_sentiment,
        'post_sentiment': post_sentiment,
        'pct_change': pct_change
    })

impact_df = pd.DataFrame(impact_results)

# ----------------------------
# Keyword Trends Analysis
# ----------------------------
# Define keywords to track in the 'clean_post_title'
keywords = ['doge department', 'elon musk', 'federal freeze', 'remote work']
for keyword in keywords:
    col_name = f'contains_{keyword.replace(" ", "_")}'
    combined_df[col_name] = combined_df['clean_post_title'].str.contains(keyword, case=False, na=False)

# ----------------------------
# Engagement Metrics Analysis
# ----------------------------
if 'post_upvote_ratio' in combined_df.columns and 'num_comments' in combined_df.columns:
    combined_df['engagement'] = combined_df['post_upvote_ratio'] * combined_df['num_comments']
    engagement_correlation = combined_df[['overall_sentiment', 'engagement']].corr().iloc[0, 1]
else:
    engagement_correlation = np.nan

# ----------------------------
# Prepare Daily Time Series Data
# ----------------------------
daily_sentiment = combined_df.set_index('post_datetime').resample('D')['overall_sentiment'].mean().reset_index()

# ----------------------------
# Save All Analysis Results to a Single Excel File with Multiple Sheets
# ----------------------------
with pd.ExcelWriter('sentiment_analysis_results.xlsx') as writer:
    combined_df.to_excel(writer, sheet_name='Raw Data', index=False)
    impact_df.to_excel(writer, sheet_name='Executive Order Impact', index=False)
    daily_sentiment.to_excel(writer, sheet_name='Daily Sentiment', index=False)
    pd.DataFrame({'engagement_correlation': [engagement_correlation]}).to_excel(
        writer, sheet_name='Engagement Metrics', index=False
    )

print("Analysis complete. Results saved to 'sentiment_analysis_results.xlsx'")

