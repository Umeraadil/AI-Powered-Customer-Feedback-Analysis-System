"""
Topic Modeling Module for AI-Powered Customer Feedback Analysis System

This module provides topic modeling capabilities to identify key themes
and topics in customer feedback using various techniques including LDA,
NMF, and keyword extraction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
from collections import Counter, defaultdict

# Machine learning imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicModeler:
    """
    A comprehensive topic modeling class supporting multiple approaches.
    """
    
    def __init__(self, n_topics: int = 5, random_state: int = 42):
        """
        Initialize the topic modeler.
        
        Args:
            n_topics (int): Number of topics to extract
            random_state (int): Random state for reproducibility
        """
        self.n_topics = n_topics
        self.random_state = random_state
        self.models = {}
        self.vectorizers = {}
        self.is_fitted = False
        self.topic_labels = {}
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        self.stop_words = set(stopwords.words('english'))
        
        logger.info(f"TopicModeler initialized with {n_topics} topics")
    
    def preprocess_for_topics(self, texts: List[str], 
                             min_word_length: int = 3,
                             custom_stopwords: Optional[List[str]] = None) -> List[str]:
        """
        Preprocess texts specifically for topic modeling.
        
        Args:
            texts (List[str]): List of texts to preprocess
            min_word_length (int): Minimum word length to keep
            custom_stopwords (Optional[List[str]]): Additional stopwords
            
        Returns:
            List[str]: Preprocessed texts
        """
        all_stopwords = self.stop_words.copy()
        if custom_stopwords:
            all_stopwords.update(custom_stopwords)
        
        processed_texts = []
        
        for text in texts:
            if not isinstance(text, str):
                processed_texts.append("")
                continue
            
            # Tokenize and filter
            tokens = word_tokenize(text.lower())
            
            # Keep only alphabetic tokens of sufficient length
            filtered_tokens = [
                token for token in tokens 
                if token.isalpha() and len(token) >= min_word_length and token not in all_stopwords
            ]
            
            # Join back to string
            processed_text = ' '.join(filtered_tokens)
            processed_texts.append(processed_text)
        
        # Remove empty texts
        processed_texts = [text for text in processed_texts if text.strip()]
        
        logger.info(f"Preprocessed {len(processed_texts)} texts for topic modeling")
        return processed_texts
    
    def fit_lda_model(self, texts: List[str], max_features: int = 1000) -> Dict[str, Any]:
        """
        Fit Latent Dirichlet Allocation (LDA) model.
        
        Args:
            texts (List[str]): Preprocessed texts
            max_features (int): Maximum number of features for vectorization
            
        Returns:
            Dict: Model fitting results
        """
        # Create count vectorizer (LDA works better with raw token counts)
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only words with 3+ characters
        )
        
        # Fit vectorizer and transform texts
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # Fit LDA model
        lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=100,
            learning_method='batch'
        )
        
        lda_model.fit(doc_term_matrix)
        
        # Store models
        self.models['lda'] = lda_model
        self.vectorizers['lda'] = vectorizer
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = self._extract_lda_topics(lda_model, feature_names)
        
        # Calculate perplexity and log-likelihood
        perplexity = lda_model.perplexity(doc_term_matrix)
        log_likelihood = lda_model.score(doc_term_matrix)
        
        logger.info(f"LDA model fitted. Perplexity: {perplexity:.2f}")
        
        return {
            'model_type': 'lda',
            'topics': topics,
            'perplexity': perplexity,
            'log_likelihood': log_likelihood,
            'n_features': len(feature_names)
        }
    
    def fit_nmf_model(self, texts: List[str], max_features: int = 1000) -> Dict[str, Any]:
        """
        Fit Non-negative Matrix Factorization (NMF) model.
        
        Args:
            texts (List[str]): Preprocessed texts
            max_features (int): Maximum number of features for vectorization
            
        Returns:
            Dict: Model fitting results
        """
        # Create TF-IDF vectorizer (NMF works better with TF-IDF)
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        
        # Fit vectorizer and transform texts
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # Fit NMF model
        nmf_model = NMF(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=200,
            alpha=0.1,
            l1_ratio=0.5
        )
        
        nmf_model.fit(doc_term_matrix)
        
        # Store models
        self.models['nmf'] = nmf_model
        self.vectorizers['nmf'] = vectorizer
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = self._extract_nmf_topics(nmf_model, feature_names)
        
        # Calculate reconstruction error
        reconstruction_error = nmf_model.reconstruction_err_
        
        logger.info(f"NMF model fitted. Reconstruction error: {reconstruction_error:.4f}")
        
        return {
            'model_type': 'nmf',
            'topics': topics,
            'reconstruction_error': reconstruction_error,
            'n_features': len(feature_names)
        }
    
    def fit_clustering_model(self, texts: List[str], max_features: int = 1000) -> Dict[str, Any]:
        """
        Fit K-means clustering model for topic discovery.
        
        Args:
            texts (List[str]): Preprocessed texts
            max_features (int): Maximum number of features for vectorization
            
        Returns:
            Dict: Model fitting results
        """
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        
        # Fit vectorizer and transform texts
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # Fit K-means model
        kmeans_model = KMeans(
            n_clusters=self.n_topics,
            random_state=self.random_state,
            max_iter=300,
            n_init=10
        )
        
        cluster_labels = kmeans_model.fit_predict(doc_term_matrix)
        
        # Store models
        self.models['kmeans'] = kmeans_model
        self.vectorizers['kmeans'] = vectorizer
        
        # Extract topics from cluster centers
        feature_names = vectorizer.get_feature_names_out()
        topics = self._extract_kmeans_topics(kmeans_model, feature_names)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(doc_term_matrix, cluster_labels)
        
        logger.info(f"K-means model fitted. Silhouette score: {silhouette_avg:.4f}")
        
        return {
            'model_type': 'kmeans',
            'topics': topics,
            'silhouette_score': silhouette_avg,
            'cluster_labels': cluster_labels.tolist(),
            'n_features': len(feature_names)
        }
    
    def _extract_lda_topics(self, model, feature_names: np.ndarray, 
                           top_words: int = 10) -> List[Dict[str, Any]]:
        """Extract topics from LDA model."""
        topics = []
        
        for topic_idx, topic in enumerate(model.components_):
            # Get top words for this topic
            top_word_indices = topic.argsort()[-top_words:][::-1]
            top_words_list = [feature_names[i] for i in top_word_indices]
            top_weights = [topic[i] for i in top_word_indices]
            
            # Generate topic label
            topic_label = self._generate_topic_label(top_words_list[:3])
            
            topics.append({
                'topic_id': topic_idx,
                'label': topic_label,
                'top_words': top_words_list,
                'word_weights': top_weights,
                'word_weight_pairs': list(zip(top_words_list, top_weights))
            })
        
        return topics
    
    def _extract_nmf_topics(self, model, feature_names: np.ndarray,
                           top_words: int = 10) -> List[Dict[str, Any]]:
        """Extract topics from NMF model."""
        topics = []
        
        for topic_idx, topic in enumerate(model.components_):
            # Get top words for this topic
            top_word_indices = topic.argsort()[-top_words:][::-1]
            top_words_list = [feature_names[i] for i in top_word_indices]
            top_weights = [topic[i] for i in top_word_indices]
            
            # Generate topic label
            topic_label = self._generate_topic_label(top_words_list[:3])
            
            topics.append({
                'topic_id': topic_idx,
                'label': topic_label,
                'top_words': top_words_list,
                'word_weights': top_weights,
                'word_weight_pairs': list(zip(top_words_list, top_weights))
            })
        
        return topics
    
    def _extract_kmeans_topics(self, model, feature_names: np.ndarray,
                              top_words: int = 10) -> List[Dict[str, Any]]:
        """Extract topics from K-means cluster centers."""
        topics = []
        
        for topic_idx, center in enumerate(model.cluster_centers_):
            # Get top words for this cluster
            top_word_indices = center.argsort()[-top_words:][::-1]
            top_words_list = [feature_names[i] for i in top_word_indices]
            top_weights = [center[i] for i in top_word_indices]
            
            # Generate topic label
            topic_label = self._generate_topic_label(top_words_list[:3])
            
            topics.append({
                'topic_id': topic_idx,
                'label': topic_label,
                'top_words': top_words_list,
                'word_weights': top_weights,
                'word_weight_pairs': list(zip(top_words_list, top_weights))
            })
        
        return topics
    
    def _generate_topic_label(self, top_words: List[str]) -> str:
        """Generate a human-readable label for a topic."""
        return " & ".join(top_words[:3]).title()
    
    def fit_all_models(self, texts: List[str], max_features: int = 1000) -> Dict[str, Any]:
        """
        Fit all available topic modeling approaches.
        
        Args:
            texts (List[str]): Preprocessed texts
            max_features (int): Maximum number of features
            
        Returns:
            Dict: Results from all models
        """
        if len(texts) < self.n_topics:
            raise ValueError(f"Need at least {self.n_topics} texts for {self.n_topics} topics")
        
        results = {}
        
        # Fit LDA model
        try:
            lda_results = self.fit_lda_model(texts, max_features)
            results['lda'] = lda_results
        except Exception as e:
            logger.error(f"Error fitting LDA model: {str(e)}")
        
        # Fit NMF model
        try:
            nmf_results = self.fit_nmf_model(texts, max_features)
            results['nmf'] = nmf_results
        except Exception as e:
            logger.error(f"Error fitting NMF model: {str(e)}")
        
        # Fit K-means model
        try:
            kmeans_results = self.fit_clustering_model(texts, max_features)
            results['kmeans'] = kmeans_results
        except Exception as e:
            logger.error(f"Error fitting K-means model: {str(e)}")
        
        self.is_fitted = True
        logger.info(f"Fitted {len(results)} topic models successfully")
        
        return results
    
    def predict_topics(self, texts: List[str], model_type: str = 'lda') -> List[Dict[str, Any]]:
        """
        Predict topics for new texts.
        
        Args:
            texts (List[str]): Texts to predict topics for
            model_type (str): Type of model to use ('lda', 'nmf', 'kmeans')
            
        Returns:
            List[Dict]: Topic predictions for each text
        """
        if not self.is_fitted or model_type not in self.models:
            raise ValueError(f"Model {model_type} not fitted")
        
        model = self.models[model_type]
        vectorizer = self.vectorizers[model_type]
        
        # Transform texts
        doc_term_matrix = vectorizer.transform(texts)
        
        predictions = []
        
        if model_type == 'lda':
            # Get topic probabilities
            topic_probs = model.transform(doc_term_matrix)
            
            for i, probs in enumerate(topic_probs):
                dominant_topic = np.argmax(probs)
                predictions.append({
                    'text_index': i,
                    'dominant_topic': int(dominant_topic),
                    'topic_probabilities': probs.tolist(),
                    'confidence': float(np.max(probs))
                })
        
        elif model_type == 'nmf':
            # Get topic weights
            topic_weights = model.transform(doc_term_matrix)
            
            for i, weights in enumerate(topic_weights):
                dominant_topic = np.argmax(weights)
                # Normalize weights to probabilities
                probs = weights / np.sum(weights) if np.sum(weights) > 0 else weights
                
                predictions.append({
                    'text_index': i,
                    'dominant_topic': int(dominant_topic),
                    'topic_probabilities': probs.tolist(),
                    'confidence': float(np.max(probs))
                })
        
        elif model_type == 'kmeans':
            # Get cluster assignments
            cluster_labels = model.predict(doc_term_matrix)
            
            # Calculate distances to centroids for confidence
            distances = model.transform(doc_term_matrix)
            
            for i, (label, dist_row) in enumerate(zip(cluster_labels, distances)):
                # Convert distances to probabilities (inverse distance)
                inv_distances = 1 / (dist_row + 1e-8)
                probs = inv_distances / np.sum(inv_distances)
                
                predictions.append({
                    'text_index': i,
                    'dominant_topic': int(label),
                    'topic_probabilities': probs.tolist(),
                    'confidence': float(np.max(probs))
                })
        
        return predictions
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'feedback_text',
                         model_type: str = 'lda') -> pd.DataFrame:
        """
        Analyze topics for all texts in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing texts
            text_column (str): Name of the text column
            model_type (str): Type of model to use
            
        Returns:
            pd.DataFrame: DataFrame with topic analysis results
        """
        df_result = df.copy()
        
        # Get texts and preprocess
        texts = df_result[text_column].astype(str).tolist()
        processed_texts = self.preprocess_for_topics(texts)
        
        # If not fitted, fit the model first
        if not self.is_fitted:
            logger.info("Model not fitted. Fitting model first...")
            self.fit_all_models(processed_texts)
        
        # Predict topics
        predictions = self.predict_topics(processed_texts, model_type)
        
        # Add results to DataFrame
        df_result['dominant_topic'] = [p['dominant_topic'] for p in predictions]
        df_result['topic_confidence'] = [p['confidence'] for p in predictions]
        
        # Add topic labels if available
        if model_type in self.models and hasattr(self, 'topic_labels'):
            topic_results = getattr(self, f'{model_type}_results', {})
            if 'topics' in topic_results:
                topic_label_map = {t['topic_id']: t['label'] for t in topic_results['topics']}
                df_result['topic_label'] = df_result['dominant_topic'].map(topic_label_map)
        
        logger.info(f"Analyzed topics for {len(df_result)} texts using {model_type}")
        
        return df_result
    
    def get_topic_distribution(self, df: pd.DataFrame,
                              topic_column: str = 'dominant_topic') -> Dict[str, Any]:
        """
        Get topic distribution statistics.
        
        Args:
            df (pd.DataFrame): DataFrame with topic predictions
            topic_column (str): Name of the topic column
            
        Returns:
            Dict: Topic distribution statistics
        """
        if topic_column not in df.columns:
            raise ValueError(f"Column '{topic_column}' not found in DataFrame")
        
        topic_counts = df[topic_column].value_counts().sort_index()
        total_count = len(df)
        
        distribution = {
            'counts': topic_counts.to_dict(),
            'percentages': (topic_counts / total_count * 100).round(2).to_dict(),
            'total_samples': total_count
        }
        
        # Calculate average confidence if available
        if 'topic_confidence' in df.columns:
            distribution['average_confidence'] = df['topic_confidence'].mean()
            distribution['confidence_by_topic'] = df.groupby(topic_column)['topic_confidence'].mean().to_dict()
        
        return distribution
    
    def extract_keywords(self, texts: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords from texts using TF-IDF.
        
        Args:
            texts (List[str]): List of texts
            top_k (int): Number of top keywords to return
            
        Returns:
            List[Tuple[str, float]]: List of (keyword, score) tuples
        """
        # Preprocess texts
        processed_texts = self.preprocess_for_topics(texts)
        
        if not processed_texts:
            return []
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate mean TF-IDF scores
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Get top keywords
        top_indices = mean_scores.argsort()[-top_k:][::-1]
        keywords = [(feature_names[i], mean_scores[i]) for i in top_indices]
        
        return keywords
    
    def save_models(self, filepath: str):
        """
        Save fitted models to file.
        
        Args:
            filepath (str): Path to save the models
        """
        if not self.is_fitted:
            raise ValueError("No fitted models to save")
        
        model_data = {
            'models': self.models,
            'vectorizers': self.vectorizers,
            'n_topics': self.n_topics,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'topic_labels': self.topic_labels
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """
        Load fitted models from file.
        
        Args:
            filepath (str): Path to load the models from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.vectorizers = model_data['vectorizers']
        self.n_topics = model_data['n_topics']
        self.random_state = model_data['random_state']
        self.is_fitted = model_data['is_fitted']
        self.topic_labels = model_data.get('topic_labels', {})
        
        logger.info(f"Models loaded from {filepath}")


def create_sample_feedback_data():
    """Create sample feedback data for testing."""
    feedback_data = [
        "The product quality is excellent and delivery was fast",
        "Customer service was very helpful and responsive",
        "Great value for money, highly recommend this product",
        "The website is easy to use and navigation is smooth",
        "Fast shipping and good packaging, product arrived safely",
        "Poor customer service, waited too long for response",
        "Product quality is not as expected, very disappointed",
        "Expensive for what you get, not worth the price",
        "Website is confusing and hard to find what I need",
        "Slow delivery and product was damaged during shipping",
        "Love the design and functionality of this product",
        "Easy to use interface and great user experience",
        "Excellent build quality and attention to detail",
        "Quick checkout process and secure payment options",
        "Outstanding customer support team, very knowledgeable"
    ]
    
    return feedback_data


if __name__ == "__main__":
    # Create sample data
    sample_texts = create_sample_feedback_data()
    
    print(f"Sample data: {len(sample_texts)} feedback entries")
    print("First few entries:")
    for i, text in enumerate(sample_texts[:3]):
        print(f"{i+1}: {text}")
    print()
    
    # Initialize topic modeler
    topic_modeler = TopicModeler(n_topics=3, random_state=42)
    
    # Preprocess texts
    processed_texts = topic_modeler.preprocess_for_topics(sample_texts)
    print(f"Preprocessed {len(processed_texts)} texts")
    print()
    
    # Fit all models
    results = topic_modeler.fit_all_models(processed_texts, max_features=100)
    
    # Display results
    for model_type, result in results.items():
        print(f"\n{model_type.upper()} Results:")
        print(f"Number of topics: {len(result['topics'])}")
        
        for topic in result['topics']:
            print(f"Topic {topic['topic_id']}: {topic['label']}")
            print(f"Top words: {', '.join(topic['top_words'][:5])}")
        print()
    
    # Test topic prediction
    test_texts = [
        "Great customer service and fast delivery",
        "Poor product quality and expensive price",
        "Easy to use website with good design"
    ]
    
    print("Topic Predictions for Test Texts:")
    predictions = topic_modeler.predict_topics(test_texts, model_type='lda')
    
    for text, pred in zip(test_texts, predictions):
        print(f"Text: {text}")
        print(f"Dominant topic: {pred['dominant_topic']} (confidence: {pred['confidence']:.3f})")
        print()
    
    # Extract keywords
    keywords = topic_modeler.extract_keywords(sample_texts, top_k=10)
    print("Top Keywords:")
    for keyword, score in keywords:
        print(f"{keyword}: {score:.4f}")
    
    # Test DataFrame analysis
    df = pd.DataFrame({'feedback_text': sample_texts})
    analyzed_df = topic_modeler.analyze_dataframe(df, model_type='lda')
    
    print(f"\nDataFrame Analysis Results:")
    print(analyzed_df[['feedback_text', 'dominant_topic', 'topic_confidence']].head())
    
    # Get topic distribution
    distribution = topic_modeler.get_topic_distribution(analyzed_df)
    print(f"\nTopic Distribution: {distribution}")

