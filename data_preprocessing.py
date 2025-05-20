import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DataPreprocessor:
    def __init__(self, file_path):
        """
        Initialize the data preprocessor with the path to the dataset
        
        Args:
            file_path (str): Path to the CSV file containing reviews
        """
        self.file_path = file_path
        self.data = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """
        Load the data from CSV file
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        self.data = pd.read_csv(self.file_path)
        print(f"Loaded data with shape: {self.data.shape}")
        return self.data
    
    def clean_text(self, text):
        """
        Clean text data by removing special characters, numbers, and converting to lowercase
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        # Join tokens back into a string
        return ' '.join(cleaned_tokens)
    
    def preprocess(self):
        """
        Preprocess the data by:
        1. Removing missing values
        2. Cleaning text
        3. Creating sentiment labels based on rating
        4. Balancing the dataset (optional)
        
        Returns:
            pandas.DataFrame: Preprocessed data
        """
        if self.data is None:
            self.load_data()
        
        # Make a copy to avoid modifying the original data
        df = self.data.copy()
        
        # Clean text in review column
        print("Cleaning text data...")
        df['cleaned_review'] = df['review'].apply(self.clean_text)
        
        # Create sentiment labels based on rating
        print("Creating sentiment labels...")
        df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 
                                            else ('negative' if x <= 2 else 'neutral'))
        
        # Convert sentiment to numerical values for modeling
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        df['sentiment_score'] = df['sentiment'].map(sentiment_map)
        
        # Drop rows with empty cleaned reviews
        df = df[df['cleaned_review'].str.strip() != '']
        
        print(f"Preprocessing complete. Final shape: {df.shape}")
        
        return df
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets
        
        Args:
            df (pandas.DataFrame): Preprocessed data
            test_size (float): Proportion of the dataset to include in the test split
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X = df['cleaned_review']
        y = df['sentiment_score']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, df, output_path="processed_data.csv"):
        """
        Save the preprocessed data to a CSV file
        
        Args:
            df (pandas.DataFrame): Preprocessed data
            output_path (str): Path to save the processed data
            
        Returns:
            str: Path to the saved file
        """
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        return output_path

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor("chatgpt_reviews - chatgpt_reviews.csv")
    processed_data = preprocessor.preprocess()
    
    # Save the processed data
    preprocessor.save_processed_data(processed_data, "processed_chatgpt_reviews.csv")
    
    # Split the data for model training
    X_train, X_test, y_train, y_test = preprocessor.split_data(processed_data)
    
    # Display sample of preprocessed data
    print("\nSample of preprocessed data:")
    print(processed_data[['cleaned_review', 'sentiment']].head())
