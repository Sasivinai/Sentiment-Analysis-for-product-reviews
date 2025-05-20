import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor

class SentimentModel:
    def __init__(self):
        """Initialize the sentiment analysis model"""
        self.models = {
            'naive_bayes': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('classifier', MultinomialNB())
            ]),
            'logistic_regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('classifier', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            'random_forest': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'svm': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('classifier', LinearSVC(random_state=42))
            ])
        }
        self.best_model = None
        self.best_model_name = None
        self.sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train multiple models and evaluate their performance
        
        Args:
            X_train (pandas.Series): Training text data
            X_test (pandas.Series): Testing text data
            y_train (pandas.Series): Training labels
            y_test (pandas.Series): Testing labels
            
        Returns:
            dict: Results for each model
        """
        results = {}
        best_accuracy = 0
        
        for name, model in self.models.items():
            print(f"\nTraining {name} model...")
            model.fit(X_train, y_train)
            
            # Predict on test set
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=self.sentiment_labels.values(), output_dict=True)
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'classification_report': report,
                'predictions': y_pred
            }
            
            print(f"{name} accuracy: {accuracy:.4f}")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} with accuracy: {best_accuracy:.4f}")
        return results
    
    def plot_confusion_matrix(self, y_test, y_pred, title="Confusion Matrix"):
        """
        Plot confusion matrix for model evaluation
        
        Args:
            y_test (pandas.Series): True labels
            y_pred (numpy.ndarray): Predicted labels
            title (str): Title for the plot
        """
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.sentiment_labels.values(),
                    yticklabels=self.sentiment_labels.values())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def predict_sentiment(self, text, preprocessor):
        """
        Predict sentiment for a given text
        
        Args:
            text (str): Text to predict sentiment for
            preprocessor (DataPreprocessor): Preprocessor instance to clean the text
            
        Returns:
            str: Predicted sentiment (negative, neutral, positive)
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
        
        # Clean the text
        cleaned_text = preprocessor.clean_text(text)
        
        # Predict
        prediction = self.best_model.predict([cleaned_text])[0]
        
        # Map to sentiment label
        return self.sentiment_labels[prediction]
    
    def save_model(self, path="sentiment_model.pkl"):
        """
        Save the best model to a file
        
        Args:
            path (str): Path to save the model
            
        Returns:
            str: Path to the saved model
        """
        if self.best_model is None:
            raise ValueError("No model to save. Train a model first.")
        
        joblib.dump(self.best_model, path)
        print(f"Model saved to {path}")
        return path
    
    @staticmethod
    def load_model(path="sentiment_model.pkl"):
        """
        Load a saved model
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            SentimentModel: Loaded model
        """
        model_instance = SentimentModel()
        model_instance.best_model = joblib.load(path)
        return model_instance

if __name__ == "__main__":
    # Load and preprocess data
    preprocessor = DataPreprocessor("chatgpt_reviews - chatgpt_reviews.csv")
    processed_data = preprocessor.preprocess()
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(processed_data)
    
    # Train and evaluate models
    sentiment_model = SentimentModel()
    results = sentiment_model.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Plot confusion matrix for the best model
    y_pred = results[sentiment_model.best_model_name]['predictions']
    sentiment_model.plot_confusion_matrix(y_test, y_pred, 
                                         title=f"Confusion Matrix - {sentiment_model.best_model_name}")
    
    # Save the best model
    sentiment_model.save_model("best_sentiment_model.pkl")
    
    # Example prediction
    print("\nExample prediction:")
    example_text = "This tool is fantastic and has helped me so much with my work"
    sentiment = sentiment_model.predict_sentiment(example_text, preprocessor)
    print(f"Text: {example_text}")
    print(f"Predicted sentiment: {sentiment}")
