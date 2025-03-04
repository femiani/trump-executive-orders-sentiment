Trump Executive Orders Sentiment Analysis (2025)

📌 Project Overview
This project analyzes public sentiment regarding Trumps election win and key executive orders issued by Trump in 2025 by scraping and processing Reddit posts. The analysis focuses on:
- Executive Order Impact: Comparing public sentiment 3 days before and after key executive orders, such as the Federal Hiring Freeze, Remote Work Ban, and Doge Department Lunch.
- Keyword Trends: Tracking the prevalence of phrases (e.g., "doge department", "Elon Musk") in post titles.
- Engagement Metrics: Examining the relationship between sentiment and engagement (computed as post upvote ratio multiplied by the number of comments).
- Temporal Analysis: Aggregating sentiment on a daily basis to observe trends over time.

The ultimate goal is to create compelling visualizations that reveal how public sentiment shifts in response to political events.

🛠️ Methodology

Data Collection
- Scraping:  
  Reddit posts were scraped using the [PRAW](https://praw.readthedocs.io/) library from subreddits like `politics`, `conservative`, and `news`.  
  - Parameters:  
    - Keywords: "trump", "executive order", "federal"  
    - Date Range: From `2025-01-20` onward  
  - The raw scraped data was saved as CSV files.

Data Cleaning
- Techniques:  
  The raw data underwent extensive cleaning:
  - Removal of URLs and special characters  
  - Expansion of contractions (using the `contractions` package)  
  - Conversion to lowercase and removal of stopwords (using NLTK)  
  - Lemmatization (using NLTK's WordNetLemmatizer)  
- Outcome:  
  Cleaned data was saved in files such as `election_posts_clean.csv` and `executive_posts_clean.csv`.

Sentiment Analysis
- Tool:  
  NLTK’s VADER sentiment analyzer was used to compute sentiment scores (compound, positive, negative, neutral) for textual fields.
- Overall Sentiment:  
  The overall sentiment score for each post is computed as the average of compound scores from the post title and comment body.
- Temporal Aggregation:  
  Daily average sentiment scores were calculated to assess trends over time.

Advanced Analyses
1. Executive Order Impact:  
   Posts within a 3-day window before and after each executive order were compared. The analysis calculates:
   - Average sentiment before (pre_sentiment) and after (post_sentiment) each order  
    
2. Keyword Trends:
   Boolean indicators track the presence of specific keywords (e.g., "doge department", "Elon Musk") in post titles. These indicators are then analyzed across sentiment bins to reveal trends.

3. Engagement Metrics:  
   Engagement is defined as the product of the post’s upvote ratio and its number of comments. The analysis explores the correlation between overall sentiment and engagement.

Visualizations:

Visualizations were created in Python using Matplotlib and Seaborn. Key charts include:
- Executive Order Impact Chart:  
  - A bar chart comparing pre- and post-executive order sentiment.
  - Saved as `executive_impact.png`.
  
- Keyword Trends Visualization: 
  - A bar chart showing the average frequency of key phrases in post titles.
  - A heatmap illustrating keyword frequency across different sentiment bins.
  - Saved as `keyword_heatmap.png`.

- Engagement Metrics Analysis:  
  - A scatter plot that examines the relationship between overall sentiment and engagement, with a trend line indicating the correlation.
  - Saved as `engagement_analysis.png`.

- Daily Sentiment Time Series:  
  - A line chart plotting the daily average overall sentiment with markers for executive orders.
  - Saved as `daily_sentiment.png`.
 
- Distrubution of overall sentiment
  - Displays the distribution of overall sentiment for the election win.
  - Saved as 'election_sentiment_histogram'
 
- Daily average sentiment
  - Tracks daily sentiment trends to show how public opinion evolves over time.
  - Saved as 'election_sentiment_timeseries'

🗂️ File Structure:

trump project/
├── data/                   # Raw and cleaned datasets
├── analysis/               # data processing and analysis
├── visuals/                # Generated charts (PNG files)
├── src/                    # Source code for scraping, cleaning, and visualization
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

▶️ Reproduction Guide:
- Install Dependencies:
pip install -r requirements.txt

- Run the Analysis Pipeline:
Execute the main analysis script:
python main.py --scrape --clean --analyze

- Generate Visualizations by running:
python visualize.py --format png --dpi 300

- Review the Results:
The outputs (e.g., sentiment_analysis_results.xlsx and PNG files) will be in the data/ and visuals/ folders, respectively.

Interpretation Guide

- Understanding the Visuals:
Sentiment Scores:
- Positive (>0.05): Supportive discussions
- Neutral (-0.05 to 0.05): Fact-based comments
- Negative (<-0.05): Critical perspectives

Engagement Formula
- Engagement = (Upvotes / Total Votes) × Number of Comments
Measures both popularity and discussion intensity

Statistical Significance:

- Used Welch's t-test (p < 0.05) for pre/post comparisons
- 95% confidence intervals shown in bar chart error bars

💡 Conclusion and Future Work:

This project reveals significant public sentiment shifts around key executive orders and highlights the interplay between sentiment, keyword trends, and engagement. Future work may include:
- Expanding data sources to include Twitter or other platforms.
- Implementing advanced machine learning techniques for sentiment classification.
- Creating interactive dashboards for deeper analysis.
