import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import nltk
from data_preprocessing import DataPreprocessor

# Download necessary NLTK resources if not already downloaded
nltk.download('punkt')

class ChatGPTReviewAnalyzer:
    def __init__(self, file_path):
        """
        Initialize the ChatGPT review analyzer
        
        Args:
            file_path (str): Path to the CSV file containing reviews
        """
        self.file_path = file_path
        self.preprocessor = DataPreprocessor(file_path)
        self.raw_data = None
        self.processed_data = None
        
    def load_and_process_data(self):
        """
        Load and preprocess the data
        
        Returns:
            pandas.DataFrame: Processed data
        """
        self.raw_data = self.preprocessor.load_data()
        self.processed_data = self.preprocessor.preprocess()
        return self.processed_data
    
    def plot_rating_distribution(self, save_path='rating_distribution.png'):
        """
        Plot the distribution of ratings
        
        Args:
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x='rating', data=self.raw_data, palette='viridis')
        plt.title('Distribution of Ratings', fontsize=16)
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Rating distribution plot saved to {save_path}")
    
    def plot_sentiment_distribution(self, save_path='sentiment_distribution.png'):
        """
        Plot the distribution of sentiments
        
        Args:
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x='sentiment', data=self.processed_data, 
                      order=['negative', 'neutral', 'positive'],
                      palette={'negative': 'red', 'neutral': 'gray', 'positive': 'green'})
        plt.title('Distribution of Sentiments', fontsize=16)
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Sentiment distribution plot saved to {save_path}")
    
    def plot_reviews_by_platform(self, save_path='platform_distribution.png'):
        """
        Plot the count of reviews by platform
        
        Args:
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        platform_counts = self.raw_data['platform'].value_counts()
        plt.pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=sns.color_palette('pastel', len(platform_counts)))
        plt.title('Reviews by Platform', fontsize=16)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Platform distribution plot saved to {save_path}")
    
    def plot_sentiment_by_title(self, top_n=10, save_path='sentiment_by_title.png'):
        """
        Plot sentiment distribution for top review titles
        
        Args:
            top_n (int): Number of top titles to display
            save_path (str): Path to save the plot
        """
        # Get the top review titles
        title_counts = self.raw_data['title'].value_counts().head(top_n)
        top_titles = title_counts.index.tolist()
        
        # Filter data for top titles
        filtered_data = self.processed_data[self.processed_data['title'].isin(top_titles)]
        
        # Create a pivot table for plotting
        pivot_table = pd.crosstab(filtered_data['title'], filtered_data['sentiment'])
        
        # Plot
        plt.figure(figsize=(12, 8))
        pivot_table.plot(kind='bar', stacked=True, 
                         color=['red', 'gray', 'green'], 
                         ax=plt.gca())
        plt.title(f'Sentiment Distribution for Top {top_n} Review Titles', fontsize=16)
        plt.xlabel('Review Title', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Sentiment by title plot saved to {save_path}")
    
    def plot_word_cloud(self, sentiment=None, save_path='wordcloud.png'):
        """
        Generate a word cloud from the reviews
        
        Args:
            sentiment (str, optional): Filter by sentiment ('positive', 'neutral', 'negative')
            save_path (str): Path to save the word cloud image
        """
        # Filter data by sentiment if specified
        if sentiment:
            text_data = ' '.join(self.processed_data[self.processed_data['sentiment'] == sentiment]['cleaned_review'])
            title = f'{sentiment.capitalize()} Reviews Word Cloud'
        else:
            text_data = ' '.join(self.processed_data['cleaned_review'])
            title = 'All Reviews Word Cloud'
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            max_words=200, 
            contour_width=3, 
            contour_color='steelblue'
        ).generate(text_data)
        
        # Plot
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Word cloud saved to {save_path}")
    
    def plot_rating_by_language(self, save_path='rating_by_language.png'):
        """
        Plot average rating by language
        
        Args:
            save_path (str): Path to save the plot
        """
        # Calculate average rating by language
        avg_ratings = self.raw_data.groupby('language')['rating'].mean().sort_values(ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        avg_ratings.plot(kind='bar', color=sns.color_palette('viridis', len(avg_ratings)))
        plt.title('Average Rating by Language', fontsize=16)
        plt.xlabel('Language', fontsize=12)
        plt.ylabel('Average Rating', fontsize=12)
        plt.ylim(0, 5)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Rating by language plot saved to {save_path}")
    
    def plot_review_trends_over_time(self, save_path='review_trends.png'):
        """
        Plot review counts and sentiment trends over time
        
        Args:
            save_path (str): Path to save the plot
        """
        # Convert date to datetime
        self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
        self.processed_data['date'] = pd.to_datetime(self.processed_data['date'])
        
        # Group by month and calculate review counts
        monthly_counts = self.raw_data.resample('M', on='date').size()
        
        # Calculate monthly average rating
        monthly_rating = self.raw_data.resample('M', on='date')['rating'].mean()
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot review counts
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Number of Reviews', color='blue', fontsize=12)
        ax1.plot(monthly_counts.index, monthly_counts.values, color='blue', marker='o')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create second y-axis for average rating
        ax2 = ax1.twinx()
        ax2.set_ylabel('Average Rating', color='red', fontsize=12)
        ax2.plot(monthly_rating.index, monthly_rating.values, color='red', marker='x')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 5)
        
        # Set title and adjust layout
        plt.title('Review Count and Average Rating by Month', fontsize=16)
        fig.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Review trends plot saved to {save_path}")
    
    def generate_all_visualizations(self):
        """
        Generate all visualizations for the exploratory analysis
        """
        # Load and process data if not already done
        if self.processed_data is None:
            self.load_and_process_data()
        
        # Generate all plots
        self.plot_rating_distribution()
        self.plot_sentiment_distribution()
        self.plot_reviews_by_platform()
        self.plot_sentiment_by_title()
        
        # Generate word clouds for different sentiments
        self.plot_word_cloud(sentiment='positive', save_path='positive_wordcloud.png')
        self.plot_word_cloud(sentiment='neutral', save_path='neutral_wordcloud.png')
        self.plot_word_cloud(sentiment='negative', save_path='negative_wordcloud.png')
        
        self.plot_rating_by_language()
        self.plot_review_trends_over_time()
        
        print("All visualizations generated successfully!")


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ChatGPTReviewAnalyzer("chatgpt_reviews - chatgpt_reviews.csv")
    
    # Generate all visualizations
    analyzer.generate_all_visualizations()
