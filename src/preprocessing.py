"""
Data Preprocessing Module for AI-Powered Customer Feedback Analysis System

This module handles the preprocessing of customer feedback data including
text cleaning, normalization, tokenization, and feature extraction.
"""

import pandas as pd
import numpy as np
import re
import string
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackPreprocessor:
    """
    A comprehensive class for preprocessing customer feedback data.
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the preprocessor.
        
        Args:
            language (str): Language for stopwords and processing
        """
        self.language = language
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker')
        
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
        
        self.stop_words = set(stopwords.words(language))
        
        # Common contractions mapping
        self.contractions = {
            "ain't": "are not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is",
            "let's": "let us", "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they'd": "they would", "they'll": "they will",
            "they're": "they are", "they've": "they have", "we'd": "we would",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what's": "what is", "where's": "where is", "who's": "who is",
            "won't": "will not", "wouldn't": "would not", "you'd": "you would",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }
        
        logger.info("FeedbackPreprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits (keep basic punctuation)
        text = re.sub(r'[^a-zA-Z\s\.\!\?\,\;\:]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str, custom_stopwords: Optional[List[str]] = None) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text (str): Text to process
            custom_stopwords (Optional[List[str]]): Additional stopwords to remove
            
        Returns:
            str: Text with stopwords removed
        """
        tokens = word_tokenize(text)
        
        # Combine default and custom stopwords
        all_stopwords = self.stop_words.copy()
        if custom_stopwords:
            all_stopwords.update(custom_stopwords)
        
        filtered_tokens = [token for token in tokens if token.lower() not in all_stopwords]
        
        return ' '.join(filtered_tokens)
    
    def stem_text(self, text: str) -> str:
        """
        Apply stemming to text.
        
        Args:
            text (str): Text to stem
            
        Returns:
            str: Stemmed text
        """
        tokens = word_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Apply lemmatization to text.
        
        Args:
            text (str): Text to lemmatize
            
        Returns:
            str: Lemmatized text
        """
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)
    
    def extract_features(self, text: str) -> Dict[str, any]:
        """
        Extract various features from text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Dictionary containing extracted features
        """
        features = {}
        
        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(word_tokenize(text))
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in word_tokenize(text)]) if word_tokenize(text) else 0
        
        # Punctuation analysis
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['punctuation_ratio'] = sum([1 for char in text if char in string.punctuation]) / len(text) if text else 0
        
        # Uppercase analysis
        features['uppercase_ratio'] = sum([1 for char in text if char.isupper()]) / len(text) if text else 0
        
        # Part-of-speech analysis
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        pos_counts = Counter([tag for word, tag in pos_tags])
        
        features['noun_count'] = pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)
        features['verb_count'] = pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)
        features['adjective_count'] = pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0)
        features['adverb_count'] = pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + pos_counts.get('RBS', 0)
        
        return features
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'feedback_text',
                           steps: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply preprocessing steps to a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing feedback data
            text_column (str): Name of the column containing text data
            steps (Optional[List[str]]): List of preprocessing steps to apply
            
        Returns:
            pd.DataFrame: DataFrame with preprocessed text and additional features
        """
        if steps is None:
            steps = ['clean', 'remove_stopwords', 'lemmatize']
        
        df_processed = df.copy()
        
        # Ensure text column exists
        if text_column not in df_processed.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame")
        
        # Store original text
        df_processed['original_text'] = df_processed[text_column]
        
        # Apply preprocessing steps
        for step in steps:
            if step == 'clean':
                logger.info("Applying text cleaning...")
                df_processed[text_column] = df_processed[text_column].apply(self.clean_text)
            
            elif step == 'remove_stopwords':
                logger.info("Removing stopwords...")
                df_processed[text_column] = df_processed[text_column].apply(self.remove_stopwords)
            
            elif step == 'stem':
                logger.info("Applying stemming...")
                df_processed[text_column] = df_processed[text_column].apply(self.stem_text)
            
            elif step == 'lemmatize':
                logger.info("Applying lemmatization...")
                df_processed[text_column] = df_processed[text_column].apply(self.lemmatize_text)
            
            else:
                logger.warning(f"Unknown preprocessing step: {step}")
        
        # Extract features
        logger.info("Extracting text features...")
        features_list = df_processed['original_text'].apply(self.extract_features).tolist()
        features_df = pd.DataFrame(features_list)
        
        # Combine with original DataFrame
        df_processed = pd.concat([df_processed, features_df], axis=1)
        
        # Remove empty texts
        initial_count = len(df_processed)
        df_processed = df_processed[df_processed[text_column].str.strip() != '']
        final_count = len(df_processed)
        
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} empty texts after preprocessing")
        
        logger.info(f"Preprocessing completed. Final dataset shape: {df_processed.shape}")
        
        return df_processed
    
    def create_tfidf_features(self, texts: List[str], max_features: int = 1000,
                             ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[np.ndarray, TfidfVectorizer]:
        """
        Create TF-IDF features from text data.
        
        Args:
            texts (List[str]): List of texts to vectorize
            max_features (int): Maximum number of features
            ngram_range (Tuple[int, int]): Range of n-grams to consider
            
        Returns:
            Tuple[np.ndarray, TfidfVectorizer]: TF-IDF matrix and fitted vectorizer
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        logger.info(f"Created TF-IDF features with shape: {tfidf_matrix.shape}")
        
        return tfidf_matrix.toarray(), vectorizer
    
    def create_bow_features(self, texts: List[str], max_features: int = 1000,
                           ngram_range: Tuple[int, int] = (1, 1)) -> Tuple[np.ndarray, CountVectorizer]:
        """
        Create Bag of Words features from text data.
        
        Args:
            texts (List[str]): List of texts to vectorize
            max_features (int): Maximum number of features
            ngram_range (Tuple[int, int]): Range of n-grams to consider
            
        Returns:
            Tuple[np.ndarray, CountVectorizer]: BoW matrix and fitted vectorizer
        """
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        bow_matrix = vectorizer.fit_transform(texts)
        
        logger.info(f"Created BoW features with shape: {bow_matrix.shape}")
        
        return bow_matrix.toarray(), vectorizer
    
    def get_preprocessing_summary(self, df_original: pd.DataFrame, df_processed: pd.DataFrame,
                                text_column: str = 'feedback_text') -> Dict[str, any]:
        """
        Generate a summary of the preprocessing results.
        
        Args:
            df_original (pd.DataFrame): Original DataFrame
            df_processed (pd.DataFrame): Processed DataFrame
            text_column (str): Name of the text column
            
        Returns:
            Dict: Summary statistics
        """
        summary = {}
        
        # Basic statistics
        summary['original_count'] = len(df_original)
        summary['processed_count'] = len(df_processed)
        summary['removed_count'] = summary['original_count'] - summary['processed_count']
        
        # Text length statistics
        original_lengths = df_original[text_column].str.len()
        processed_lengths = df_processed[text_column].str.len()
        
        summary['original_avg_length'] = original_lengths.mean()
        summary['processed_avg_length'] = processed_lengths.mean()
        summary['length_reduction_ratio'] = 1 - (summary['processed_avg_length'] / summary['original_avg_length'])
        
        # Word count statistics
        original_word_counts = df_original[text_column].str.split().str.len()
        processed_word_counts = df_processed[text_column].str.split().str.len()
        
        summary['original_avg_words'] = original_word_counts.mean()
        summary['processed_avg_words'] = processed_word_counts.mean()
        summary['word_reduction_ratio'] = 1 - (summary['processed_avg_words'] / summary['original_avg_words'])
        
        # Feature statistics (if available)
        feature_columns = ['char_count', 'word_count', 'sentence_count', 'noun_count', 'verb_count']
        available_features = [col for col in feature_columns if col in df_processed.columns]
        
        if available_features:
            summary['extracted_features'] = available_features
            summary['feature_stats'] = df_processed[available_features].describe().to_dict()
        
        logger.info("Preprocessing summary generated")
        
        return summary


# Example usage and testing functions
def create_sample_noisy_data():
    """Create sample noisy data for testing preprocessing."""
    noisy_feedback = [
        "GREAT product!!! I LOVE it so much!!! https://example.com",
        "The service was terrible... I'm very disappointed :( contact@company.com",
        "It's okay, nothing special but not bad either. 123-456-7890",
        "Excellent customer support! They're amazing and resolved my issue quickly!!!",
        "Poor quality for the price... wouldn't recommend it to anyone. #disappointed",
        "   This has extra    whitespace   and    weird formatting   ",
        "Can't believe how good this is! Won't disappoint you at all!",
        "<p>This has HTML tags</p> and other <b>formatting</b> issues"
    ]
    
    return pd.DataFrame({'feedback_text': noisy_feedback})


if __name__ == "__main__":
    # Create sample noisy data
    df_sample = create_sample_noisy_data()
    
    print("Original Data:")
    for i, text in enumerate(df_sample['feedback_text']):
        print(f"{i+1}: {text}")
    print()
    
    # Initialize preprocessor
    preprocessor = FeedbackPreprocessor()
    
    # Test preprocessing
    df_processed = preprocessor.preprocess_dataframe(
        df_sample,
        steps=['clean', 'remove_stopwords', 'lemmatize']
    )
    
    print("Processed Data:")
    for i, (original, processed) in enumerate(zip(df_processed['original_text'], df_processed['feedback_text'])):
        print(f"{i+1}: Original: {original}")
        print(f"   Processed: {processed}")
        print()
    
    # Show extracted features
    print("Sample Features:")
    feature_columns = ['char_count', 'word_count', 'sentence_count', 'exclamation_count', 'question_count']
    print(df_processed[feature_columns].head())
    
    # Generate preprocessing summary
    summary = preprocessor.get_preprocessing_summary(df_sample, df_processed)
    print("\nPreprocessing Summary:")
    for key, value in summary.items():
        if key != 'feature_stats':
            print(f"{key}: {value}")
    
    # Test TF-IDF features
    tfidf_matrix, tfidf_vectorizer = preprocessor.create_tfidf_features(
        df_processed['feedback_text'].tolist(),
        max_features=50
    )
    
    print(f"\nTF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print("Top TF-IDF Features:")
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print(feature_names[:10])

