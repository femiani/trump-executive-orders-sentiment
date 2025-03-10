import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

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
