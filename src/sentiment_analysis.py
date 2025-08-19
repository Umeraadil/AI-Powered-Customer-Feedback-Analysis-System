"""
Sentiment Analysis Module for AI-Powered Customer Feedback Analysis System

This module provides sentiment analysis capabilities for customer feedback
using various approaches including lexicon-based methods and machine learning models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
import pickle
import os
from pathlib import Path

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# NLTK imports for lexicon-based analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    A comprehensive sentiment analysis class supporting multiple approaches.
    """
    
    def __init__(self, model_type: str = 'lexicon'):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_type (str): Type of model to use ('lexicon', 'ml', 'ensemble')
        """
        self.model_type = model_type
        self.models = {}
        self.vectorizers = {}
        self.is_trained = False
        
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        # Initialize VADER sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Define sentiment labels
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        self.label_to_int = {label: i for i, label in enumerate(self.sentiment_labels)}
        self.int_to_label = {i: label for i, label in enumerate(self.sentiment_labels)}
        
        logger.info(f"SentimentAnalyzer initialized with model type: {model_type}")
    
    def analyze_sentiment_lexicon(self, text: str) -> Dict[str, Union[str, float, Dict]]:
        """
        Analyze sentiment using lexicon-based approach (VADER).
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Sentiment analysis results
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0, 'compound': 0.0}
            }
        
        # Get VADER scores
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine overall sentiment based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
            confidence = compound
        elif compound <= -0.05:
            sentiment = 'negative'
            confidence = abs(compound)
        else:
            sentiment = 'neutral'
            confidence = 1 - abs(compound)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': scores
        }
    
    def prepare_training_data(self, df: pd.DataFrame, text_column: str = 'feedback_text',
                            label_column: str = 'sentiment') -> Tuple[List[str], List[str]]:
        """
        Prepare training data for machine learning models.
        
        Args:
            df (pd.DataFrame): DataFrame containing training data
            text_column (str): Name of the text column
            label_column (str): Name of the label column
            
        Returns:
            Tuple[List[str], List[str]]: Texts and labels
        """
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].astype(str).tolist()
        
        # Validate labels
        valid_labels = set(self.sentiment_labels)
        invalid_labels = set(labels) - valid_labels
        
        if invalid_labels:
            logger.warning(f"Found invalid labels: {invalid_labels}. Valid labels are: {valid_labels}")
            # Filter out invalid labels
            valid_indices = [i for i, label in enumerate(labels) if label in valid_labels]
            texts = [texts[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
        
        logger.info(f"Prepared {len(texts)} training samples")
        return texts, labels
    
    def train_ml_models(self, texts: List[str], labels: List[str],
                       test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Train machine learning models for sentiment analysis.
        
        Args:
            texts (List[str]): Training texts
            labels (List[str]): Training labels
            test_size (float): Proportion of data to use for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            Dict[str, float]: Model performance scores
        """
        if len(texts) < 10:
            raise ValueError("Need at least 10 samples for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Define models to train
        models_to_train = {
            'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'naive_bayes': MultinomialNB(),
            'svm': SVC(random_state=random_state, probability=True),
            'random_forest': RandomForestClassifier(random_state=random_state, n_estimators=100)
        }
        
        # Train models
        performance_scores = {}
        
        for model_name, model in models_to_train.items():
            logger.info(f"Training {model_name}...")
            
            # Create pipeline with TF-IDF vectorizer
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model and performance
            self.models[model_name] = pipeline
            performance_scores[model_name] = accuracy
            
            logger.info(f"{model_name} accuracy: {accuracy:.4f}")
        
        # Select best model
        best_model_name = max(performance_scores, key=performance_scores.get)
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        self.is_trained = True
        
        logger.info(f"Best model: {best_model_name} with accuracy: {performance_scores[best_model_name]:.4f}")
        
        return performance_scores
    
    def analyze_sentiment_ml(self, text: str) -> Dict[str, Union[str, float, Dict]]:
        """
        Analyze sentiment using trained machine learning model.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Sentiment analysis results
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_ml_models() first.")
        
        if not isinstance(text, str) or not text.strip():
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'probabilities': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
            }
        
        # Predict sentiment
        prediction = self.best_model.predict([text])[0]
        probabilities = self.best_model.predict_proba([text])[0]
        
        # Get probability scores for each class
        prob_dict = {
            label: float(prob) for label, prob in zip(self.sentiment_labels, probabilities)
        }
        
        confidence = float(max(probabilities))
        
        return {
            'sentiment': prediction,
            'confidence': confidence,
            'probabilities': prob_dict,
            'model_used': self.best_model_name
        }
    
    def analyze_sentiment_ensemble(self, text: str) -> Dict[str, Union[str, float, Dict]]:
        """
        Analyze sentiment using ensemble of lexicon and ML approaches.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Sentiment analysis results
        """
        # Get lexicon-based result
        lexicon_result = self.analyze_sentiment_lexicon(text)
        
        # Get ML-based result if model is trained
        if self.is_trained:
            ml_result = self.analyze_sentiment_ml(text)
            
            # Combine results (weighted average)
            lexicon_weight = 0.3
            ml_weight = 0.7
            
            # Convert sentiments to scores for averaging
            sentiment_to_score = {'negative': -1, 'neutral': 0, 'positive': 1}
            score_to_sentiment = {-1: 'negative', 0: 'neutral', 1: 'positive'}
            
            lexicon_score = sentiment_to_score[lexicon_result['sentiment']]
            ml_score = sentiment_to_score[ml_result['sentiment']]
            
            combined_score = lexicon_weight * lexicon_score + ml_weight * ml_score
            
            # Determine final sentiment
            if combined_score > 0.1:
                final_sentiment = 'positive'
            elif combined_score < -0.1:
                final_sentiment = 'negative'
            else:
                final_sentiment = 'neutral'
            
            # Calculate combined confidence
            combined_confidence = (lexicon_weight * lexicon_result['confidence'] + 
                                 ml_weight * ml_result['confidence'])
            
            return {
                'sentiment': final_sentiment,
                'confidence': combined_confidence,
                'lexicon_result': lexicon_result,
                'ml_result': ml_result,
                'ensemble_score': combined_score
            }
        else:
            # Fall back to lexicon-based analysis
            logger.warning("ML model not trained, using lexicon-based analysis only")
            return lexicon_result
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Dict]: List of sentiment analysis results
        """
        results = []
        
        for text in texts:
            if self.model_type == 'lexicon':
                result = self.analyze_sentiment_lexicon(text)
            elif self.model_type == 'ml':
                result = self.analyze_sentiment_ml(text)
            elif self.model_type == 'ensemble':
                result = self.analyze_sentiment_ensemble(text)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            results.append(result)
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'feedback_text') -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing texts
            text_column (str): Name of the text column
            
        Returns:
            pd.DataFrame: DataFrame with sentiment analysis results
        """
        df_result = df.copy()
        
        # Analyze sentiment for all texts
        texts = df_result[text_column].astype(str).tolist()
        results = self.analyze_batch(texts)
        
        # Add results to DataFrame
        df_result['predicted_sentiment'] = [r['sentiment'] for r in results]
        df_result['sentiment_confidence'] = [r['confidence'] for r in results]
        
        # Add detailed scores if available
        if self.model_type == 'lexicon':
            df_result['vader_compound'] = [r['scores']['compound'] for r in results]
            df_result['vader_positive'] = [r['scores']['pos'] for r in results]
            df_result['vader_negative'] = [r['scores']['neg'] for r in results]
            df_result['vader_neutral'] = [r['scores']['neu'] for r in results]
        
        elif self.model_type == 'ml' and self.is_trained:
            df_result['prob_positive'] = [r['probabilities']['positive'] for r in results]
            df_result['prob_negative'] = [r['probabilities']['negative'] for r in results]
            df_result['prob_neutral'] = [r['probabilities']['neutral'] for r in results]
        
        logger.info(f"Analyzed sentiment for {len(df_result)} texts")
        
        return df_result
    
    def get_sentiment_distribution(self, df: pd.DataFrame, 
                                 sentiment_column: str = 'predicted_sentiment') -> Dict[str, any]:
        """
        Get sentiment distribution statistics.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment predictions
            sentiment_column (str): Name of the sentiment column
            
        Returns:
            Dict: Sentiment distribution statistics
        """
        if sentiment_column not in df.columns:
            raise ValueError(f"Column '{sentiment_column}' not found in DataFrame")
        
        sentiment_counts = df[sentiment_column].value_counts()
        total_count = len(df)
        
        distribution = {
            'counts': sentiment_counts.to_dict(),
            'percentages': (sentiment_counts / total_count * 100).round(2).to_dict(),
            'total_samples': total_count
        }
        
        # Calculate average confidence if available
        if 'sentiment_confidence' in df.columns:
            distribution['average_confidence'] = df['sentiment_confidence'].mean()
        
        return distribution
    
    def save_model(self, filepath: str):
        """
        Save trained models to file.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'models': self.models,
            'best_model_name': self.best_model_name,
            'model_type': self.model_type,
            'sentiment_labels': self.sentiment_labels,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained models from file.
        
        Args:
            filepath (str): Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.best_model_name = model_data['best_model_name']
        self.best_model = self.models[self.best_model_name]
        self.model_type = model_data['model_type']
        self.sentiment_labels = model_data['sentiment_labels']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")


def create_sample_training_data():
    """Create sample training data for testing."""
    sample_data = [
        ("I love this product! It's amazing and works perfectly.", "positive"),
        ("Great quality and fast delivery. Highly recommend!", "positive"),
        ("Excellent customer service. Very satisfied.", "positive"),
        ("This product is okay, nothing special.", "neutral"),
        ("Average quality, meets basic expectations.", "neutral"),
        ("It's fine, not great but not terrible either.", "neutral"),
        ("Terrible product! Waste of money.", "negative"),
        ("Poor quality and bad customer service.", "negative"),
        ("I hate this product. It doesn't work at all.", "negative"),
        ("Outstanding product! Exceeded my expectations completely.", "positive"),
        ("Good value for money. Would buy again.", "positive"),
        ("The product is decent but could be better.", "neutral"),
        ("Not impressed. Expected much more for the price.", "negative"),
        ("Awful experience. Would not recommend to anyone.", "negative"),
        ("Perfect! Exactly what I was looking for.", "positive")
    ]
    
    df = pd.DataFrame(sample_data, columns=['feedback_text', 'sentiment'])
    return df


if __name__ == "__main__":
    # Create sample training data
    training_df = create_sample_training_data()
    
    print("Sample Training Data:")
    print(training_df.head())
    print(f"Data shape: {training_df.shape}")
    print()
    
    # Test lexicon-based analysis
    print("Testing Lexicon-based Analysis:")
    lexicon_analyzer = SentimentAnalyzer(model_type='lexicon')
    
    test_texts = [
        "I love this product!",
        "This is terrible.",
        "It's okay, nothing special."
    ]
    
    for text in test_texts:
        result = lexicon_analyzer.analyze_sentiment_lexicon(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        print()
    
    # Test ML-based analysis
    print("Testing ML-based Analysis:")
    ml_analyzer = SentimentAnalyzer(model_type='ml')
    
    # Prepare training data
    texts, labels = ml_analyzer.prepare_training_data(training_df)
    
    # Train models
    performance = ml_analyzer.train_ml_models(texts, labels)
    print("Model Performance:")
    for model_name, accuracy in performance.items():
        print(f"{model_name}: {accuracy:.4f}")
    print()
    
    # Test predictions
    for text in test_texts:
        result = ml_analyzer.analyze_sentiment_ml(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        print(f"Model used: {result['model_used']}")
        print()
    
    # Test batch analysis
    print("Testing Batch Analysis:")
    batch_results = ml_analyzer.analyze_batch(test_texts)
    for text, result in zip(test_texts, batch_results):
        print(f"{text} -> {result['sentiment']}")
    
    # Test DataFrame analysis
    print("\nTesting DataFrame Analysis:")
    analyzed_df = ml_analyzer.analyze_dataframe(training_df)
    print(analyzed_df[['feedback_text', 'sentiment', 'predicted_sentiment', 'sentiment_confidence']].head())
    
    # Get sentiment distribution
    distribution = ml_analyzer.get_sentiment_distribution(analyzed_df)
    print(f"\nSentiment Distribution: {distribution}")

