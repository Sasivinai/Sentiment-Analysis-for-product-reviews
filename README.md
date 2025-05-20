# ChatGPT Review Sentiment Analysis

This project implements a complete Natural Language Processing (NLP) system for sentiment analysis of ChatGPT reviews. The application analyzes review text to determine whether sentiments are positive, neutral, or negative, and provides visualizations of the results.

## Features

- **Text Preprocessing**: Cleaning, tokenization, lemmatization, and stopword removal
- **Sentiment Analysis**: Machine learning models to classify sentiments 
- **Interactive Web Interface**: Streamlit-based UI for real-time sentiment analysis
- **Data Visualization**: Interactive charts and dashboards to explore the dataset
- **Multi-Model Comparison**: Training and evaluation of multiple ML models

## Project Structure

- `data_preprocessing.py`: Data loading, cleaning, and preprocessing
- `model_training.py`: ML model training, evaluation, and persistence
- `exploratory_analysis.py`: Statistical analysis and visualization scripts
- `streamlit_app.py`: Interactive web application
- `requirements.txt`: Required Python dependencies

## Dataset

The dataset contains ChatGPT reviews with:
- Review text and title
- Ratings (1-5 scale)
- User information
- Platform details (Web/Mobile)
- Language and location information
- Version information

## Setup and Installation

1. **Clone the repository or download the project files**

2. **Install the required Python packages**:
   ```
   pip install -r requirements.txt
   ```

3. **Run the data preprocessing**:
   ```
   python data_preprocessing.py
   ```

4. **Train the sentiment analysis models**:
   ```
   python model_training.py
   ```

5. **Generate visualizations (optional)**:
   ```
   python exploratory_analysis.py
   ```

6. **Launch the Streamlit web application**:
   ```
   streamlit run streamlit_app.py
   ```

## Web Application

The Streamlit application provides:
- A sentiment analysis tool for any text input
- Interactive dashboards with dataset insights
- Word clouds showing most frequent terms by sentiment
- Rating distributions and analysis
- Platform and language analysis

## Model Information

The system trains multiple models and selects the best one:
- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**

Feature extraction is done using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

## Example Usage

1. Open the Streamlit web app
2. Navigate to the "Sentiment Analysis" page
3. Enter or paste review text
4. Click "Analyze Sentiment" to get the prediction
5. Explore the "Dashboard" page for dataset insights

## Requirements

- Python 3.7+
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- Matplotlib
- Seaborn
- Plotly
- WordCloud

## License

This project is created for educational purposes.

## Acknowledgments

- Dataset: ChatGPT reviews
- NLP techniques based on modern sentiment analysis approaches
