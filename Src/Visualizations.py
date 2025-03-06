import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

# Load Cleaned Election Posts Data
df_election = pd.read_csv("election_posts_clean.csv")

plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_analysis_data(excel_path='sentiment_analysis_results.xlsx'):
    """Load data from Excel analysis output"""
    return {
        'combined': pd.read_excel(excel_path, sheet_name='Raw Data'),
        'impact': pd.read_excel(excel_path, sheet_name='Executive Order Impact'),
        'daily': pd.read_excel(excel_path, sheet_name='Daily Sentiment'),
        'engagement': pd.read_excel(excel_path, sheet_name='Engagement Metrics')
    }

def plot_executive_impact(impact_df):
    """Visualize pre/post sentiment for executive orders"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot with error margins
    impact_df.plot(x='executive_order', y=['pre_sentiment', 'post_sentiment'],
                   kind='bar', ax=ax, edgecolor='black', capsize=4)
    ax.set_title("Sentiment Impact of Executive Orders")
    ax.set_ylabel("Average Sentiment Score")
    ax.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('executive_impact.png')
    plt.close()

# Calculate Overall Sentiment 
def get_sentiment(text):
    if not isinstance(text, str):
        text = ""
    return sia.polarity_scores(text)['compound']

# If overall_sentiment is not present, calculate it:
if 'overall_sentiment' not in df_election.columns:
    df_election['sentiment_title'] = df_election['clean_post_title'].apply(get_sentiment)
    df_election['sentiment_comment'] = df_election['clean_comment_body'].apply(get_sentiment)
    # Compute overall sentiment as the average of available sentiment scores
    df_election['overall_sentiment'] = df_election[['sentiment_title', 'sentiment_comment']].mean(axis=1)

def plot_daily_sentiment(daily_df, impact_df):
    """Time series plot with executive order markers"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(daily_df['post_datetime'], daily_df['overall_sentiment'],
            marker='o', markersize=3, linestyle='-', linewidth=1)
    
    # Add vertical lines for executive orders
    for _, row in impact_df.iterrows():
        ax.axvline(row['order_date'], color='red', linestyle='--', alpha=0.7)
        ax.text(row['order_date'], ax.get_ylim()[1]*0.95, row['executive_order'],
                rotation=45, ha='right', va='top', fontsize=8)
    
    ax.set_title("Daily Sentiment Trend")
    ax.set_ylabel("Average Sentiment Score")
    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('daily_sentiment.png')
    plt.close()

def plot_engagement_correlation(combined_df, corr_value):
    """Scatter plot of sentiment vs engagement"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.regplot(x='overall_sentiment', y='engagement', data=combined_df,
                scatter_kws={'alpha':0.3, 's':20}, line_kws={'color':'darkred'}, ax=ax)
    ax.set_title(f"Sentiment vs Engagement (r = {corr_value:.2f})")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Engagement (Upvotes × Comments)")
    plt.tight_layout()
    plt.savefig('engagement_correlation.png')
    plt.close()

def plot_keyword_frequency(combined_df):
    """Heatmap of keyword mentions by sentiment"""
    keywords = [col for col in combined_df.columns if 'contains_' in col]
    
    if not keywords:
        print("No keyword columns found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create sentiment bins
    combined_df['sentiment_bin'] = pd.cut(combined_df['overall_sentiment'],
                                          bins=[-1, -0.5, 0, 0.5, 1],
                                          labels=['Very Negative', 'Negative', 'Neutral', 'Positive'])
    
    # Calculate frequency matrix
    freq_matrix = combined_df.groupby('sentiment_bin')[keywords].mean().T
    
    sns.heatmap(freq_matrix*100, annot=True, fmt=".1f", cmap="YlGnBu",
                ax=ax, cbar_kws={'label': 'Percentage %'})
    ax.set_title("Keyword Frequency by Sentiment Level")
    ax.set_yticklabels([col.replace('contains_', '').replace('_', ' ') for col in keywords])
    plt.tight_layout()
    plt.savefig('keyword_heatmap.png')
    plt.close()

    # Histogram of Overall Sentiment
plt.figure(figsize=(10,6))
sns.histplot(df_election['overall_sentiment'], bins=30, kde=True)
plt.title("Distribution of Overall Sentiment in Election Posts")
plt.xlabel("Overall Sentiment (Compound Score)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("election_sentiment_histogram.png", dpi=300, bbox_inches="tight")
plt.show()

# Daily Average Sentiment Time Series
daily_sentiment = df_election.set_index('post_datetime').resample('D')['overall_sentiment'].mean().reset_index()

plt.figure(figsize=(12,6))
plt.plot(daily_sentiment['post_datetime'], daily_sentiment['overall_sentiment'],
         marker='o', linestyle='-', linewidth=1)
plt.title("Daily Average Sentiment in Election Posts")
plt.xlabel("Date")
plt.ylabel("Average Overall Sentiment")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("election_sentiment_timeseries.png", dpi=300, bbox_inches="tight")
plt.show()

def main():
    # Load data from analysis output
    data = load_analysis_data()
    
    # Convert date columns
    data['impact']['order_date'] = pd.to_datetime(data['impact']['order_date'])
    data['daily']['post_datetime'] = pd.to_datetime(data['daily']['post_datetime'])
    
    # Generate visualizations
    plot_executive_impact(data['impact'])
    plot_daily_sentiment(data['daily'], data['impact'])
    plot_engagement_correlation(data['combined'], data['engagement'].iloc[0,0])
    plot_keyword_frequency(data['combined'])
    
    print("Visualizations saved as PNG files")

if __name__ == "__main__":
    main()