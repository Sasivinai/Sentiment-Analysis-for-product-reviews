import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from data_preprocessing import DataPreprocessor
from model_training import SentimentModel

# Set page configuration
st.set_page_config(
    page_title="ChatGPT Review Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize preprocessor
@st.cache_resource
def load_preprocessor():
    return DataPreprocessor("chatgpt_reviews - chatgpt_reviews.csv")

preprocessor = load_preprocessor()

# Load the model or train a new one
@st.cache_resource
def load_model():
    try:
        model = SentimentModel.load_model("best_sentiment_model.pkl")
        st.success("Successfully loaded pre-trained model")
        return model
    except:
        st.warning("Pre-trained model not found. Training a new model...")
        model = SentimentModel()
        return model

model = load_model()

# Load processed data
@st.cache_data
def load_processed_data():
    try:
        processed_data = pd.read_csv("processed_chatgpt_reviews.csv")
        st.success("Successfully loaded processed data")
        return processed_data
    except:
        st.warning("Processed data not found. Processing now...")
        processed_data = preprocessor.preprocess()
        preprocessor.save_processed_data(processed_data, "processed_chatgpt_reviews.csv")
        return processed_data

processed_data = load_processed_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sentiment Analysis", "Dashboard", "About"])

# Home page
if page == "Home":
    st.title("ChatGPT Review Sentiment Analysis")
    st.image("https://img.freepik.com/free-vector/sentiment-analysis-concept-illustration_114360-5182.jpg", width=400)
    
    st.markdown("""
    ## Welcome to the ChatGPT Review Sentiment Analysis App
    
    This application uses Natural Language Processing (NLP) techniques to analyze sentiment in ChatGPT reviews.
    
    ### Features:
    - **Sentiment Analysis**: Analyze the sentiment of any text input
    - **Interactive Dashboard**: Explore insights from the ChatGPT reviews dataset
    - **NLP Techniques**: Utilizing machine learning models and text preprocessing
    
    Use the sidebar to navigate through different sections of the application.
    """)
    
    # Display sample reviews
    st.subheader("Sample Reviews from Dataset")
    st.dataframe(processed_data[['date', 'title', 'review', 'rating', 'sentiment']].sample(5))

# Sentiment Analysis page
elif page == "Sentiment Analysis":
    st.title("Analyze Sentiment")
    st.markdown("Enter a review to analyze its sentiment:")
    
    # Text input
    text_input = st.text_area("Type or paste a review here:", height=150)
    
    # Analyze button
    if st.button("Analyze Sentiment"):
        if text_input.strip() == "":
            st.error("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                # Clean text
                cleaned_text = preprocessor.clean_text(text_input)
                
                # Predict sentiment
                sentiment = model.predict_sentiment(text_input, preprocessor)
                
                # Display results
                st.subheader("Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Text:**")
                    st.write(text_input)
                    
                with col2:
                    st.markdown("**Cleaned Text:**")
                    st.write(cleaned_text)
                
                # Display sentiment with appropriate color
                st.markdown("**Sentiment:**")
                sentiment_color = {
                    "positive": "green",
                    "neutral": "blue",
                    "negative": "red"
                }
                
                st.markdown(
                    f"<div style='background-color: {sentiment_color[sentiment]}; padding: 20px; border-radius: 5px; text-align: center;'>"
                    f"<h2 style='color: white;'>{sentiment.upper()}</h2>"
                    f"</div>",
                    unsafe_allow_html=True
                )

# Dashboard page
elif page == "Dashboard":
    st.title("ChatGPT Reviews Dashboard")
    st.markdown("""
    This dashboard provides insights into the sentiment analysis of ChatGPT reviews.
    Explore the visualizations below to understand patterns and trends in the review data.
    """)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Rating Analysis", "Platform & Language"])
    
    with tab1:
        st.header("Sentiment Distribution")
        
        # Sentiment distribution chart
        sentiment_counts = processed_data['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        fig1 = px.pie(
            sentiment_counts, 
            values='Count', 
            names='Sentiment', 
            title='Sentiment Distribution',
            color='Sentiment',
            color_discrete_map={'positive': 'green', 'neutral': 'blue', 'negative': 'red'},
            hole=0.4
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Word clouds
        st.subheader("Word Clouds by Sentiment")
        
        col1, col2, col3 = st.columns(3)
        
        # Generate word cloud functions
        @st.cache_data
        def generate_wordcloud(sentiment_type):
            # Filter data by sentiment
            text_data = ' '.join(processed_data[processed_data['sentiment'] == sentiment_type]['cleaned_review'])
            
            # Create word cloud
            wordcloud = WordCloud(
                width=400, 
                height=200, 
                background_color='white', 
                max_words=100, 
                contour_width=1, 
                contour_color='steelblue'
            ).generate(text_data)
            
            # Generate the figure
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            return fig
        
        with col1:
            st.markdown("#### Positive Reviews")
            st.pyplot(generate_wordcloud('positive'))
            
        with col2:
            st.markdown("#### Neutral Reviews")
            st.pyplot(generate_wordcloud('neutral'))
            
        with col3:
            st.markdown("#### Negative Reviews")
            st.pyplot(generate_wordcloud('negative'))
    
    with tab2:
        st.header("Rating Analysis")
        
        # Rating distribution
        rating_counts = processed_data['rating'].value_counts().reset_index()
        rating_counts.columns = ['Rating', 'Count']
        rating_counts = rating_counts.sort_values('Rating')
        
        fig2 = px.bar(
            rating_counts, 
            x='Rating', 
            y='Count', 
            title='Rating Distribution',
            color='Rating',
            labels={'Rating': 'Rating (1-5)', 'Count': 'Number of Reviews'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Rating by title
        st.subheader("Average Rating by Review Title")
        
        avg_rating_by_title = processed_data.groupby('title')['rating'].mean().reset_index()
        avg_rating_by_title = avg_rating_by_title.sort_values('rating', ascending=False).head(10)
        
        fig3 = px.bar(
            avg_rating_by_title, 
            x='rating', 
            y='title', 
            orientation='h',
            title='Average Rating by Top 10 Review Titles',
            color='rating',
            labels={'title': 'Review Title', 'rating': 'Average Rating'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.header("Platform and Language Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Platform distribution
            platform_counts = processed_data['platform'].value_counts().reset_index()
            platform_counts.columns = ['Platform', 'Count']
            
            fig4 = px.pie(
                platform_counts, 
                values='Count', 
                names='Platform',
                title='Reviews by Platform'
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            # Language distribution
            language_counts = processed_data['language'].value_counts().reset_index()
            language_counts.columns = ['Language', 'Count']
            
            fig5 = px.bar(
                language_counts, 
                x='Language', 
                y='Count',
                title='Reviews by Language',
                color='Language'
            )
            st.plotly_chart(fig5, use_container_width=True)
        
        # Average rating by language
        st.subheader("Average Rating by Language")
        
        avg_rating_by_language = processed_data.groupby('language')['rating'].mean().reset_index()
        avg_rating_by_language = avg_rating_by_language.sort_values('rating', ascending=False)
        
        fig6 = px.bar(
            avg_rating_by_language, 
            x='language', 
            y='rating',
            title='Average Rating by Language',
            color='language',
            labels={'language': 'Language', 'rating': 'Average Rating'}
        )
        st.plotly_chart(fig6, use_container_width=True)

# About page
elif page == "About":
    st.title("About This Project")
    
    st.markdown("""
    ## ChatGPT Review Sentiment Analysis
    
    This project implements Natural Language Processing (NLP) techniques to analyze sentiment in ChatGPT reviews.
    
    ### Implementation Details
    
    - **Data Preprocessing**: Text cleaning, tokenization, lemmatization, and stopword removal
    - **Feature Extraction**: TF-IDF Vectorization
    - **Machine Learning Models**: 
        - Multinomial Naive Bayes
        - Logistic Regression
        - Random Forest
        - Support Vector Machine
    - **Sentiment Classification**: Positive, Neutral, Negative
    
    ### Dataset
    
    The dataset contains reviews of ChatGPT from various users, including:
    
    - Review text
    - Ratings (1-5 scale)
    - Platform information (Web/Mobile)
    - Language
    - Location
    - Version
    
    ### Project Structure
    
    - `data_preprocessing.py`: Text cleaning and preprocessing functions
    - `model_training.py`: ML model training and evaluation
    - `exploratory_analysis.py`: Data visualization and insights
    - `streamlit_app.py`: Interactive web application
    
    Developed as part of an NLP sentiment analysis project.
    """)
    
    # Show model performance if available
    if hasattr(model, 'best_model_name') and model.best_model_name:
        st.subheader("Model Performance")
        st.write(f"Best model: {model.best_model_name}")

# Add a footer
st.markdown("---")
st.markdown("© 2025 ChatGPT Review Sentiment Analysis Project")

if __name__ == "__main__":
    # This is already running the Streamlit app
    pass
