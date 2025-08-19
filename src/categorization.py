"""
Categorization Module for AI-Powered Customer Feedback Analysis System

This module provides automatic categorization of customer feedback into
predefined categories using rule-based and machine learning approaches.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import os
import re
from collections import Counter, defaultdict

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# NLTK imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackCategorizer:
    """
    A comprehensive feedback categorization class supporting rule-based and ML approaches.
    """
    
    def __init__(self, categories: Optional[List[str]] = None):
        """
        Initialize the feedback categorizer.
        
        Args:
            categories (Optional[List[str]]): List of category names
        """
        # Default categories if none provided
        self.categories = categories or [
            'Product Quality',
            'Customer Service',
            'Pricing',
            'Delivery/Shipping',
            'Website/UI',
            'General'
        ]
        
        self.models = {}
        self.vectorizers = {}
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Rule-based categorization keywords
        self.category_keywords = self._initialize_category_keywords()
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        self.stop_words = set(stopwords.words('english'))
        
        logger.info(f"FeedbackCategorizer initialized with categories: {self.categories}")
    
    def _initialize_category_keywords(self) -> Dict[str, List[str]]:
        """Initialize keyword mappings for rule-based categorization."""
        return {
            'Product Quality': [
                'quality', 'defective', 'broken', 'durable', 'sturdy', 'flimsy',
                'material', 'build', 'construction', 'craftsmanship', 'design',
                'functionality', 'performance', 'reliable', 'unreliable',
                'excellent', 'poor', 'cheap', 'premium', 'solid', 'fragile'
            ],
            'Customer Service': [
                'service', 'support', 'staff', 'representative', 'agent',
                'helpful', 'rude', 'polite', 'professional', 'knowledgeable',
                'response', 'assistance', 'help', 'customer care', 'team',
                'friendly', 'unfriendly', 'courteous', 'attitude', 'behavior'
            ],
            'Pricing': [
                'price', 'cost', 'expensive', 'cheap', 'affordable', 'value',
                'money', 'worth', 'overpriced', 'reasonable', 'budget',
                'discount', 'deal', 'offer', 'promotion', 'sale',
                'pricing', 'fee', 'charge', 'payment', 'refund'
            ],
            'Delivery/Shipping': [
                'delivery', 'shipping', 'transport', 'arrived', 'package',
                'fast', 'slow', 'quick', 'delayed', 'on time', 'late',
                'packaging', 'box', 'damaged', 'tracking', 'courier',
                'shipment', 'dispatch', 'logistics', 'freight', 'postal'
            ],
            'Website/UI': [
                'website', 'site', 'interface', 'navigation', 'menu',
                'page', 'loading', 'slow', 'fast', 'responsive', 'mobile',
                'design', 'layout', 'user experience', 'ux', 'ui',
                'search', 'filter', 'checkout', 'cart', 'browser',
                'app', 'application', 'platform', 'system', 'online'
            ],
            'General': [
                'overall', 'experience', 'satisfied', 'disappointed',
                'recommend', 'again', 'future', 'impressed', 'happy',
                'unhappy', 'pleased', 'frustrated', 'amazing', 'terrible'
            ]
        }
    
    def add_category_keywords(self, category: str, keywords: List[str]):
        """
        Add keywords for a specific category.
        
        Args:
            category (str): Category name
            keywords (List[str]): List of keywords to add
        """
        if category not in self.categories:
            self.categories.append(category)
        
        if category not in self.category_keywords:
            self.category_keywords[category] = []
        
        self.category_keywords[category].extend(keywords)
        logger.info(f"Added {len(keywords)} keywords to category '{category}'")
    
    def categorize_rule_based(self, text: str, 
                             return_scores: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Categorize feedback using rule-based keyword matching.
        
        Args:
            text (str): Text to categorize
            return_scores (bool): Whether to return category scores
            
        Returns:
            Union[str, Dict]: Category name or detailed results
        """
        if not isinstance(text, str) or not text.strip():
            if return_scores:
                return {
                    'category': 'General',
                    'confidence': 0.0,
                    'scores': {cat: 0.0 for cat in self.categories}
                }
            return 'General'
        
        text_lower = text.lower()
        tokens = word_tokenize(text_lower)
        tokens = [token for token in tokens if token not in self.stop_words and token.isalpha()]
        
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                # Check for exact keyword match
                if keyword.lower() in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
                
                # Check for keyword in tokens (handles word boundaries better)
                if keyword.lower() in tokens:
                    score += 0.5
            
            # Normalize score by text length
            normalized_score = score / len(tokens) if tokens else 0
            category_scores[category] = {
                'raw_score': score,
                'normalized_score': normalized_score,
                'matched_keywords': matched_keywords
            }
        
        # Find category with highest score
        best_category = max(category_scores.keys(), 
                           key=lambda x: category_scores[x]['normalized_score'])
        
        best_score = category_scores[best_category]['normalized_score']
        confidence = min(best_score * 2, 1.0)  # Scale confidence
        
        if return_scores:
            return {
                'category': best_category,
                'confidence': confidence,
                'scores': {cat: data['normalized_score'] for cat, data in category_scores.items()},
                'matched_keywords': category_scores[best_category]['matched_keywords']
            }
        
        return best_category
    
    def prepare_training_data(self, df: pd.DataFrame, 
                             text_column: str = 'feedback_text',
                             category_column: str = 'category') -> Tuple[List[str], List[str]]:
        """
        Prepare training data for machine learning models.
        
        Args:
            df (pd.DataFrame): DataFrame containing training data
            text_column (str): Name of the text column
            category_column (str): Name of the category column
            
        Returns:
            Tuple[List[str], List[str]]: Texts and categories
        """
        texts = df[text_column].astype(str).tolist()
        categories = df[category_column].astype(str).tolist()
        
        # Validate categories
        valid_categories = set(self.categories)
        invalid_categories = set(categories) - valid_categories
        
        if invalid_categories:
            logger.warning(f"Found invalid categories: {invalid_categories}")
            logger.warning(f"Valid categories are: {valid_categories}")
            
            # Filter out invalid categories
            valid_indices = [i for i, cat in enumerate(categories) if cat in valid_categories]
            texts = [texts[i] for i in valid_indices]
            categories = [categories[i] for i in valid_indices]
        
        logger.info(f"Prepared {len(texts)} training samples")
        return texts, categories
    
    def train_ml_models(self, texts: List[str], categories: List[str],
                       test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Train machine learning models for categorization.
        
        Args:
            texts (List[str]): Training texts
            categories (List[str]): Training categories
            test_size (float): Proportion of data to use for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            Dict[str, float]: Model performance scores
        """
        if len(texts) < 10:
            raise ValueError("Need at least 10 samples for training")
        
        if len(set(categories)) < 2:
            raise ValueError("Need at least 2 different categories for training")
        
        # Encode labels
        encoded_categories = self.label_encoder.fit_transform(categories)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, encoded_categories, test_size=test_size, 
            random_state=random_state, stratify=encoded_categories
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
            logger.info(f"Training {model_name} for categorization...")
            
            # Create pipeline with TF-IDF vectorizer
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), 
                                        stop_words='english', lowercase=True)),
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
    
    def categorize_ml(self, text: str, 
                     return_probabilities: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Categorize feedback using trained machine learning model.
        
        Args:
            text (str): Text to categorize
            return_probabilities (bool): Whether to return category probabilities
            
        Returns:
            Union[str, Dict]: Category name or detailed results
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_ml_models() first.")
        
        if not isinstance(text, str) or not text.strip():
            if return_probabilities:
                return {
                    'category': 'General',
                    'confidence': 0.0,
                    'probabilities': {cat: 1.0/len(self.categories) for cat in self.categories}
                }
            return 'General'
        
        # Predict category
        prediction_encoded = self.best_model.predict([text])[0]
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        if return_probabilities:
            # Get probabilities
            probabilities = self.best_model.predict_proba([text])[0]
            
            # Map probabilities to category names
            prob_dict = {}
            for i, prob in enumerate(probabilities):
                category = self.label_encoder.inverse_transform([i])[0]
                prob_dict[category] = float(prob)
            
            confidence = float(max(probabilities))
            
            return {
                'category': prediction,
                'confidence': confidence,
                'probabilities': prob_dict,
                'model_used': self.best_model_name
            }
        
        return prediction
    
    def categorize_ensemble(self, text: str, 
                           rule_weight: float = 0.3,
                           ml_weight: float = 0.7) -> Dict[str, Any]:
        """
        Categorize feedback using ensemble of rule-based and ML approaches.
        
        Args:
            text (str): Text to categorize
            rule_weight (float): Weight for rule-based approach
            ml_weight (float): Weight for ML approach
            
        Returns:
            Dict: Ensemble categorization results
        """
        # Get rule-based result
        rule_result = self.categorize_rule_based(text, return_scores=True)
        
        # Get ML-based result if model is trained
        if self.is_trained:
            ml_result = self.categorize_ml(text, return_probabilities=True)
            
            # Combine results
            combined_scores = {}
            
            for category in self.categories:
                rule_score = rule_result['scores'].get(category, 0)
                ml_score = ml_result['probabilities'].get(category, 0)
                
                combined_score = rule_weight * rule_score + ml_weight * ml_score
                combined_scores[category] = combined_score
            
            # Find best category
            best_category = max(combined_scores, key=combined_scores.get)
            confidence = combined_scores[best_category]
            
            return {
                'category': best_category,
                'confidence': confidence,
                'combined_scores': combined_scores,
                'rule_result': rule_result,
                'ml_result': ml_result
            }
        else:
            # Fall back to rule-based categorization
            logger.warning("ML model not trained, using rule-based categorization only")
            return rule_result
    
    def categorize_batch(self, texts: List[str], 
                        method: str = 'ensemble') -> List[Dict[str, Any]]:
        """
        Categorize a batch of texts.
        
        Args:
            texts (List[str]): List of texts to categorize
            method (str): Method to use ('rule', 'ml', 'ensemble')
            
        Returns:
            List[Dict]: List of categorization results
        """
        results = []
        
        for text in texts:
            if method == 'rule':
                result = self.categorize_rule_based(text, return_scores=True)
            elif method == 'ml':
                if not self.is_trained:
                    raise ValueError("ML model not trained")
                result = self.categorize_ml(text, return_probabilities=True)
            elif method == 'ensemble':
                result = self.categorize_ensemble(text)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results.append(result)
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, 
                         text_column: str = 'feedback_text',
                         method: str = 'ensemble') -> pd.DataFrame:
        """
        Categorize all texts in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing texts
            text_column (str): Name of the text column
            method (str): Categorization method to use
            
        Returns:
            pd.DataFrame: DataFrame with categorization results
        """
        df_result = df.copy()
        
        # Get texts
        texts = df_result[text_column].astype(str).tolist()
        
        # Categorize texts
        results = self.categorize_batch(texts, method)
        
        # Add results to DataFrame
        df_result['predicted_category'] = [r['category'] for r in results]
        df_result['category_confidence'] = [r['confidence'] for r in results]
        
        # Add method-specific columns
        if method == 'rule':
            df_result['matched_keywords'] = [r.get('matched_keywords', []) for r in results]
        elif method == 'ml' and self.is_trained:
            df_result['model_used'] = [r.get('model_used', '') for r in results]
        
        logger.info(f"Categorized {len(df_result)} texts using {method} method")
        
        return df_result
    
    def get_category_distribution(self, df: pd.DataFrame,
                                category_column: str = 'predicted_category') -> Dict[str, Any]:
        """
        Get category distribution statistics.
        
        Args:
            df (pd.DataFrame): DataFrame with category predictions
            category_column (str): Name of the category column
            
        Returns:
            Dict: Category distribution statistics
        """
        if category_column not in df.columns:
            raise ValueError(f"Column '{category_column}' not found in DataFrame")
        
        category_counts = df[category_column].value_counts()
        total_count = len(df)
        
        distribution = {
            'counts': category_counts.to_dict(),
            'percentages': (category_counts / total_count * 100).round(2).to_dict(),
            'total_samples': total_count
        }
        
        # Calculate average confidence if available
        if 'category_confidence' in df.columns:
            distribution['average_confidence'] = df['category_confidence'].mean()
            distribution['confidence_by_category'] = df.groupby(category_column)['category_confidence'].mean().to_dict()
        
        return distribution
    
    def evaluate_model(self, df: pd.DataFrame, 
                      text_column: str = 'feedback_text',
                      true_category_column: str = 'category',
                      method: str = 'ml') -> Dict[str, Any]:
        """
        Evaluate categorization model performance.
        
        Args:
            df (pd.DataFrame): DataFrame with true categories
            text_column (str): Name of the text column
            true_category_column (str): Name of the true category column
            method (str): Method to evaluate
            
        Returns:
            Dict: Evaluation results
        """
        # Get predictions
        df_pred = self.analyze_dataframe(df, text_column, method)
        
        # Get true and predicted categories
        y_true = df[true_category_column].tolist()
        y_pred = df_pred['predicted_category'].tolist()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'method': method
        }
    
    def save_model(self, filepath: str):
        """
        Save trained models to file.
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'categories': self.categories,
            'category_keywords': self.category_keywords,
            'models': self.models,
            'vectorizers': self.vectorizers,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        
        if self.is_trained:
            model_data['best_model_name'] = self.best_model_name
        
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
        
        self.categories = model_data['categories']
        self.category_keywords = model_data['category_keywords']
        self.models = model_data['models']
        self.vectorizers = model_data.get('vectorizers', {})
        self.label_encoder = model_data['label_encoder']
        self.is_trained = model_data['is_trained']
        
        if self.is_trained:
            self.best_model_name = model_data['best_model_name']
            self.best_model = self.models[self.best_model_name]
        
        logger.info(f"Model loaded from {filepath}")


def create_sample_categorized_data():
    """Create sample categorized data for testing."""
    sample_data = [
        ("The product quality is excellent and durable", "Product Quality"),
        ("Customer service was very helpful and responsive", "Customer Service"),
        ("Too expensive for what you get", "Pricing"),
        ("Fast delivery and good packaging", "Delivery/Shipping"),
        ("Website is easy to navigate and user-friendly", "Website/UI"),
        ("Overall great experience, highly recommend", "General"),
        ("Product broke after one week, poor quality", "Product Quality"),
        ("Staff was rude and unhelpful", "Customer Service"),
        ("Great value for money", "Pricing"),
        ("Slow shipping and damaged package", "Delivery/Shipping"),
        ("App crashes frequently, needs improvement", "Website/UI"),
        ("Satisfied with my purchase", "General"),
        ("Excellent build quality and design", "Product Quality"),
        ("Quick response from support team", "Customer Service"),
        ("Reasonable price point", "Pricing")
    ]
    
    df = pd.DataFrame(sample_data, columns=['feedback_text', 'category'])
    return df


if __name__ == "__main__":
    # Create sample training data
    training_df = create_sample_categorized_data()
    
    print("Sample Training Data:")
    print(training_df.head())
    print(f"Data shape: {training_df.shape}")
    print(f"Categories: {training_df['category'].unique()}")
    print()
    
    # Initialize categorizer
    categorizer = FeedbackCategorizer()
    
    # Test rule-based categorization
    print("Testing Rule-based Categorization:")
    test_texts = [
        "The product quality is amazing",
        "Customer service was terrible",
        "Too expensive for the quality",
        "Fast delivery, arrived on time"
    ]
    
    for text in test_texts:
        result = categorizer.categorize_rule_based(text, return_scores=True)
        print(f"Text: {text}")
        print(f"Category: {result['category']} (confidence: {result['confidence']:.3f})")
        print(f"Matched keywords: {result['matched_keywords']}")
        print()
    
    # Train ML models
    print("Training ML Models:")
    texts, categories = categorizer.prepare_training_data(training_df)
    performance = categorizer.train_ml_models(texts, categories)
    
    print("Model Performance:")
    for model_name, accuracy in performance.items():
        print(f"{model_name}: {accuracy:.4f}")
    print()
    
    # Test ML categorization
    print("Testing ML Categorization:")
    for text in test_texts:
        result = categorizer.categorize_ml(text, return_probabilities=True)
        print(f"Text: {text}")
        print(f"Category: {result['category']} (confidence: {result['confidence']:.3f})")
        print(f"Model used: {result['model_used']}")
        print()
    
    # Test ensemble categorization
    print("Testing Ensemble Categorization:")
    for text in test_texts:
        result = categorizer.categorize_ensemble(text)
        print(f"Text: {text}")
        print(f"Category: {result['category']} (confidence: {result['confidence']:.3f})")
        print()
    
    # Test DataFrame analysis
    print("Testing DataFrame Analysis:")
    analyzed_df = categorizer.analyze_dataframe(training_df, method='ensemble')
    print(analyzed_df[['feedback_text', 'category', 'predicted_category', 'category_confidence']].head())
    
    # Get category distribution
    distribution = categorizer.get_category_distribution(analyzed_df)
    print(f"\nCategory Distribution: {distribution}")
    
    # Evaluate model
    evaluation = categorizer.evaluate_model(training_df, method='ml')
    print(f"\nModel Evaluation - Accuracy: {evaluation['accuracy']:.4f}")

