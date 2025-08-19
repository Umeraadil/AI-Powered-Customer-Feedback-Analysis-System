"""
Data Ingestion Module for AI-Powered Customer Feedback Analysis System

This module handles the ingestion of customer feedback data from various sources
including CSV files, JSON files, and direct API submissions.
"""

import pandas as pd
import json
import logging
from typing import List, Dict, Union, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackDataIngestion:
    """
    A class to handle ingestion of customer feedback data from multiple sources.
    """
    
    def __init__(self):
        """Initialize the data ingestion class."""
        self.supported_formats = ['csv', 'json', 'txt']
        logger.info("FeedbackDataIngestion initialized")
    
    def ingest_csv(self, file_path: str, text_column: str = 'feedback', 
                   rating_column: Optional[str] = None) -> pd.DataFrame:
        """
        Ingest feedback data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            text_column (str): Name of the column containing feedback text
            rating_column (Optional[str]): Name of the column containing ratings
            
        Returns:
            pd.DataFrame: DataFrame containing the ingested feedback data
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} records from {file_path}")
            
            # Validate required columns
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in CSV")
            
            # Standardize column names
            df = df.rename(columns={text_column: 'feedback_text'})
            if rating_column and rating_column in df.columns:
                df = df.rename(columns={rating_column: 'rating'})
            
            # Add metadata
            df['source'] = 'csv'
            df['ingestion_timestamp'] = pd.Timestamp.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting CSV file {file_path}: {str(e)}")
            raise
    
    def ingest_json(self, file_path: str, text_field: str = 'feedback',
                    rating_field: Optional[str] = None) -> pd.DataFrame:
        """
        Ingest feedback data from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            text_field (str): Name of the field containing feedback text
            rating_field (Optional[str]): Name of the field containing ratings
            
        Returns:
            pd.DataFrame: DataFrame containing the ingested feedback data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Handle both single object and array of objects
            if isinstance(data, dict):
                data = [data]
            
            df = pd.DataFrame(data)
            logger.info(f"Successfully loaded {len(df)} records from {file_path}")
            
            # Validate required fields
            if text_field not in df.columns:
                raise ValueError(f"Text field '{text_field}' not found in JSON")
            
            # Standardize column names
            df = df.rename(columns={text_field: 'feedback_text'})
            if rating_field and rating_field in df.columns:
                df = df.rename(columns={rating_field: 'rating'})
            
            # Add metadata
            df['source'] = 'json'
            df['ingestion_timestamp'] = pd.Timestamp.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting JSON file {file_path}: {str(e)}")
            raise
    
    def ingest_text_file(self, file_path: str, delimiter: str = '\n') -> pd.DataFrame:
        """
        Ingest feedback data from a plain text file.
        
        Args:
            file_path (str): Path to the text file
            delimiter (str): Delimiter to separate individual feedback entries
            
        Returns:
            pd.DataFrame: DataFrame containing the ingested feedback data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split content by delimiter and create DataFrame
            feedback_list = [text.strip() for text in content.split(delimiter) if text.strip()]
            df = pd.DataFrame({'feedback_text': feedback_list})
            
            logger.info(f"Successfully loaded {len(df)} records from {file_path}")
            
            # Add metadata
            df['source'] = 'text'
            df['ingestion_timestamp'] = pd.Timestamp.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting text file {file_path}: {str(e)}")
            raise
    
    def ingest_direct_feedback(self, feedback_data: Union[str, List[str], Dict]) -> pd.DataFrame:
        """
        Ingest feedback data directly from memory (API submissions, etc.).
        
        Args:
            feedback_data: Feedback data in various formats
            
        Returns:
            pd.DataFrame: DataFrame containing the ingested feedback data
        """
        try:
            if isinstance(feedback_data, str):
                # Single feedback string
                df = pd.DataFrame({'feedback_text': [feedback_data]})
            elif isinstance(feedback_data, list):
                # List of feedback strings
                df = pd.DataFrame({'feedback_text': feedback_data})
            elif isinstance(feedback_data, dict):
                # Dictionary with feedback data
                df = pd.DataFrame([feedback_data])
                if 'feedback_text' not in df.columns and 'feedback' in df.columns:
                    df = df.rename(columns={'feedback': 'feedback_text'})
            else:
                raise ValueError("Unsupported feedback data format")
            
            logger.info(f"Successfully ingested {len(df)} direct feedback records")
            
            # Add metadata
            df['source'] = 'direct'
            df['ingestion_timestamp'] = pd.Timestamp.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting direct feedback: {str(e)}")
            raise
    
    def batch_ingest(self, file_paths: List[str], file_types: List[str]) -> pd.DataFrame:
        """
        Ingest feedback data from multiple files in batch.
        
        Args:
            file_paths (List[str]): List of file paths
            file_types (List[str]): List of file types corresponding to each path
            
        Returns:
            pd.DataFrame: Combined DataFrame containing all ingested feedback data
        """
        if len(file_paths) != len(file_types):
            raise ValueError("Number of file paths must match number of file types")
        
        all_dataframes = []
        
        for file_path, file_type in zip(file_paths, file_types):
            try:
                if file_type.lower() == 'csv':
                    df = self.ingest_csv(file_path)
                elif file_type.lower() == 'json':
                    df = self.ingest_json(file_path)
                elif file_type.lower() == 'txt':
                    df = self.ingest_text_file(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_type}")
                    continue
                
                all_dataframes.append(df)
                
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {str(e)}")
                continue
        
        if not all_dataframes:
            raise ValueError("No files were successfully ingested")
        
        # Combine all DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"Successfully combined {len(combined_df)} records from {len(all_dataframes)} files")
        
        return combined_df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate the ingested feedback data.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            Dict: Validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': []
        }
        
        # Check if required columns exist
        if 'feedback_text' not in df.columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Missing required 'feedback_text' column")
        
        # Check for empty feedback
        if df['feedback_text'].isnull().any():
            validation_results['is_valid'] = False
            validation_results['issues'].append("Found null values in feedback_text")
        
        # Check for empty strings
        empty_feedback = df['feedback_text'].str.strip().eq('').sum()
        if empty_feedback > 0:
            validation_results['issues'].append(f"Found {empty_feedback} empty feedback entries")
        
        # Check data types
        if not df['feedback_text'].dtype == 'object':
            validation_results['issues'].append("feedback_text column should be of string type")
        
        logger.info(f"Data validation completed. Valid: {validation_results['is_valid']}")
        return validation_results


# Example usage and testing functions
def create_sample_data():
    """Create sample data for testing purposes."""
    sample_feedback = [
        "Great product! I love the quality and design.",
        "The service was terrible. Very disappointed.",
        "Average experience. Nothing special but not bad either.",
        "Excellent customer support. They resolved my issue quickly.",
        "Poor quality for the price. Would not recommend."
    ]
    
    sample_ratings = [5, 1, 3, 5, 2]
    
    # Create sample CSV
    df_csv = pd.DataFrame({
        'feedback': sample_feedback,
        'rating': sample_ratings,
        'customer_id': range(1, 6)
    })
    df_csv.to_csv('/tmp/sample_feedback.csv', index=False)
    
    # Create sample JSON
    json_data = [
        {'feedback': feedback, 'rating': rating, 'customer_id': i+1}
        for i, (feedback, rating) in enumerate(zip(sample_feedback, sample_ratings))
    ]
    
    with open('/tmp/sample_feedback.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Create sample text file
    with open('/tmp/sample_feedback.txt', 'w') as f:
        f.write('\n'.join(sample_feedback))
    
    logger.info("Sample data files created in /tmp/")


if __name__ == "__main__":
    # Create sample data for testing
    create_sample_data()
    
    # Initialize ingestion class
    ingestion = FeedbackDataIngestion()
    
    # Test CSV ingestion
    try:
        df_csv = ingestion.ingest_csv('/tmp/sample_feedback.csv', 'feedback', 'rating')
        print("CSV Ingestion Results:")
        print(df_csv.head())
        print(f"Shape: {df_csv.shape}")
        print()
    except Exception as e:
        print(f"CSV ingestion failed: {e}")
    
    # Test JSON ingestion
    try:
        df_json = ingestion.ingest_json('/tmp/sample_feedback.json', 'feedback', 'rating')
        print("JSON Ingestion Results:")
        print(df_json.head())
        print(f"Shape: {df_json.shape}")
        print()
    except Exception as e:
        print(f"JSON ingestion failed: {e}")
    
    # Test direct feedback ingestion
    try:
        direct_feedback = ["This is a test feedback", "Another test feedback"]
        df_direct = ingestion.ingest_direct_feedback(direct_feedback)
        print("Direct Ingestion Results:")
        print(df_direct.head())
        print(f"Shape: {df_direct.shape}")
        print()
    except Exception as e:
        print(f"Direct ingestion failed: {e}")

