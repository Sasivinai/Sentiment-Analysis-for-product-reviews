from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly
import plotly.express as px
from data_preprocessing import DataPreprocessor
from model_training import SentimentModel

app = Flask(__name__, template_folder='templates', static_folder='static')

# Initialize preprocessor and load model
preprocessor = DataPreprocessor("chatgpt_reviews - chatgpt_reviews.csv")
model = None

# Try to load the model, or train a new one if not available
try:
    model = SentimentModel.load_model("best_sentiment_model.pkl")
    print("Successfully loaded pre-trained model")
except:
    print("Pre-trained model not found. Please run model_training.py first.")
    model = SentimentModel()

# Load processed data for visualizations
try:
    processed_data = pd.read_csv("processed_chatgpt_reviews.csv")
    print("Successfully loaded processed data")
except:
    print("Processed data not found. Processing now...")
    processed_data = preprocessor.preprocess()
    preprocessor.save_processed_data(processed_data, "processed_chatgpt_reviews.csv")


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """API endpoint to analyze sentiment of a given text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Ensure model is loaded
        if model.best_model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Preprocess text and predict sentiment
        cleaned_text = preprocessor.clean_text(text)
        sentiment = model.predict_sentiment(text, preprocessor)
        
        # Return prediction
        return jsonify({
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard')
def dashboard():
    """Render the analytics dashboard"""
    # Create visualizations for the dashboard
    charts = create_dashboard_charts(processed_data)
    return render_template('dashboard.html', charts=charts)


def create_dashboard_charts(data):
    """
    Create charts for the dashboard
    
    Args:
        data (pandas.DataFrame): Processed data
        
    Returns:
        dict: Dictionary containing JSON representations of plotly charts
    """
    charts = {}
    
    # 1. Sentiment Distribution
    sentiment_counts = data['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig1 = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                 title='Sentiment Distribution',
                 color='Sentiment',
                 color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'})
    charts['sentiment_dist'] = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 2. Rating Distribution
    rating_counts = data['rating'].value_counts().reset_index()
    rating_counts.columns = ['Rating', 'Count']
    rating_counts = rating_counts.sort_values('Rating')
    fig2 = px.bar(rating_counts, x='Rating', y='Count', 
                 title='Rating Distribution',
                 color='Rating', 
                 labels={'Rating': 'Rating (1-5)', 'Count': 'Number of Reviews'})
    charts['rating_dist'] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 3. Platform Distribution
    platform_counts = data['platform'].value_counts().reset_index()
    platform_counts.columns = ['Platform', 'Count']
    fig3 = px.bar(platform_counts, x='Platform', y='Count',
                 title='Reviews by Platform',
                 color='Platform')
    charts['platform_dist'] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 4. Language Distribution
    language_counts = data['language'].value_counts().reset_index()
    language_counts.columns = ['Language', 'Count']
    fig4 = px.bar(language_counts, x='Language', y='Count',
                 title='Reviews by Language',
                 color='Language')
    charts['language_dist'] = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 5. Average Rating by Title
    avg_rating_by_title = data.groupby('title')['rating'].mean().reset_index()
    avg_rating_by_title = avg_rating_by_title.sort_values('rating', ascending=False).head(10)
    fig5 = px.bar(avg_rating_by_title, x='title', y='rating',
                 title='Average Rating by Top 10 Review Titles',
                 color='rating',
                 labels={'title': 'Review Title', 'rating': 'Average Rating'})
    fig5.update_layout(xaxis_tickangle=-45)
    charts['rating_by_title'] = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
    
    return charts


@app.route('/api/data')
def get_data():
    """API endpoint to get processed data for frontend visualizations"""
    try:
        # Convert to dictionary format for JSON serialization
        data_subset = processed_data[['date', 'title', 'sentiment', 'rating', 'platform', 'language']].head(100)
        data_json = data_subset.to_dict(orient='records')
        return jsonify(data_json)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, port=5000)
