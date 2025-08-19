"""
API Module for AI-Powered Customer Feedback Analysis System

This module provides RESTful API endpoints for submitting feedback,
retrieving analysis results, and managing the feedback analysis system.
"""

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import FeedbackDataIngestion
from preprocessing import FeedbackPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize components
ingestion = FeedbackDataIngestion()
preprocessor = FeedbackPreprocessor()

# Global variables for storing data (in production, use a database)
feedback_storage = []
analysis_results = {}


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        JSON response with health status
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/feedback/submit', methods=['POST'])
def submit_feedback():
    """
    Submit new customer feedback for analysis.
    
    Expected JSON payload:
    {
        "feedback_text": "Customer feedback text",
        "rating": 5 (optional),
        "customer_id": "customer123" (optional),
        "product_id": "product456" (optional),
        "metadata": {} (optional)
    }
    
    Returns:
        JSON response with submission status and feedback ID
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'feedback_text' not in data:
            return jsonify({'error': 'feedback_text is required'}), 400
        
        if not data['feedback_text'].strip():
            return jsonify({'error': 'feedback_text cannot be empty'}), 400
        
        # Create feedback entry
        feedback_entry = {
            'id': len(feedback_storage) + 1,
            'feedback_text': data['feedback_text'],
            'rating': data.get('rating'),
            'customer_id': data.get('customer_id'),
            'product_id': data.get('product_id'),
            'metadata': data.get('metadata', {}),
            'timestamp': datetime.now().isoformat(),
            'status': 'submitted'
        }
        
        # Store feedback
        feedback_storage.append(feedback_entry)
        
        logger.info(f"Feedback submitted with ID: {feedback_entry['id']}")
        
        return jsonify({
            'message': 'Feedback submitted successfully',
            'feedback_id': feedback_entry['id'],
            'status': 'submitted'
        }), 201
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/feedback/batch', methods=['POST'])
def submit_batch_feedback():
    """
    Submit multiple feedback entries at once.
    
    Expected JSON payload:
    {
        "feedback_list": [
            {
                "feedback_text": "Feedback 1",
                "rating": 5,
                ...
            },
            {
                "feedback_text": "Feedback 2",
                "rating": 3,
                ...
            }
        ]
    }
    
    Returns:
        JSON response with batch submission status
    """
    try:
        data = request.get_json()
        
        if not data or 'feedback_list' not in data:
            return jsonify({'error': 'feedback_list is required'}), 400
        
        feedback_list = data['feedback_list']
        
        if not isinstance(feedback_list, list):
            return jsonify({'error': 'feedback_list must be an array'}), 400
        
        if len(feedback_list) == 0:
            return jsonify({'error': 'feedback_list cannot be empty'}), 400
        
        submitted_ids = []
        errors = []
        
        for i, feedback_data in enumerate(feedback_list):
            try:
                if 'feedback_text' not in feedback_data or not feedback_data['feedback_text'].strip():
                    errors.append(f"Item {i}: feedback_text is required and cannot be empty")
                    continue
                
                feedback_entry = {
                    'id': len(feedback_storage) + 1,
                    'feedback_text': feedback_data['feedback_text'],
                    'rating': feedback_data.get('rating'),
                    'customer_id': feedback_data.get('customer_id'),
                    'product_id': feedback_data.get('product_id'),
                    'metadata': feedback_data.get('metadata', {}),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'submitted'
                }
                
                feedback_storage.append(feedback_entry)
                submitted_ids.append(feedback_entry['id'])
                
            except Exception as e:
                errors.append(f"Item {i}: {str(e)}")
        
        logger.info(f"Batch feedback submitted: {len(submitted_ids)} successful, {len(errors)} errors")
        
        return jsonify({
            'message': f'Batch submission completed',
            'submitted_count': len(submitted_ids),
            'submitted_ids': submitted_ids,
            'error_count': len(errors),
            'errors': errors
        }), 201 if submitted_ids else 400
        
    except Exception as e:
        logger.error(f"Error in batch feedback submission: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/feedback/list', methods=['GET'])
def list_feedback():
    """
    Retrieve list of submitted feedback with optional filtering.
    
    Query parameters:
    - limit: Maximum number of results (default: 100)
    - offset: Number of results to skip (default: 0)
    - status: Filter by status (submitted, processed, etc.)
    - customer_id: Filter by customer ID
    - product_id: Filter by product ID
    
    Returns:
        JSON response with feedback list and metadata
    """
    try:
        # Get query parameters
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        status_filter = request.args.get('status')
        customer_id_filter = request.args.get('customer_id')
        product_id_filter = request.args.get('product_id')
        
        # Apply filters
        filtered_feedback = feedback_storage.copy()
        
        if status_filter:
            filtered_feedback = [f for f in filtered_feedback if f.get('status') == status_filter]
        
        if customer_id_filter:
            filtered_feedback = [f for f in filtered_feedback if f.get('customer_id') == customer_id_filter]
        
        if product_id_filter:
            filtered_feedback = [f for f in filtered_feedback if f.get('product_id') == product_id_filter]
        
        # Apply pagination
        total_count = len(filtered_feedback)
        paginated_feedback = filtered_feedback[offset:offset + limit]
        
        return jsonify({
            'feedback': paginated_feedback,
            'total_count': total_count,
            'limit': limit,
            'offset': offset,
            'has_more': offset + limit < total_count
        })
        
    except Exception as e:
        logger.error(f"Error listing feedback: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/feedback/<int:feedback_id>', methods=['GET'])
def get_feedback(feedback_id):
    """
    Retrieve specific feedback by ID.
    
    Args:
        feedback_id (int): ID of the feedback to retrieve
        
    Returns:
        JSON response with feedback details
    """
    try:
        feedback = next((f for f in feedback_storage if f['id'] == feedback_id), None)
        
        if not feedback:
            return jsonify({'error': 'Feedback not found'}), 404
        
        return jsonify(feedback)
        
    except Exception as e:
        logger.error(f"Error retrieving feedback {feedback_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/feedback/preprocess', methods=['POST'])
def preprocess_feedback():
    """
    Preprocess submitted feedback data.
    
    Expected JSON payload:
    {
        "feedback_ids": [1, 2, 3] (optional - if not provided, processes all)
        "steps": ["clean", "remove_stopwords", "lemmatize"] (optional)
    }
    
    Returns:
        JSON response with preprocessing results
    """
    try:
        data = request.get_json() or {}
        feedback_ids = data.get('feedback_ids')
        steps = data.get('steps', ['clean', 'remove_stopwords', 'lemmatize'])
        
        # Select feedback to process
        if feedback_ids:
            feedback_to_process = [f for f in feedback_storage if f['id'] in feedback_ids]
        else:
            feedback_to_process = feedback_storage.copy()
        
        if not feedback_to_process:
            return jsonify({'error': 'No feedback found to process'}), 404
        
        # Convert to DataFrame
        df = pd.DataFrame(feedback_to_process)
        
        # Preprocess
        df_processed = preprocessor.preprocess_dataframe(df, 'feedback_text', steps)
        
        # Update feedback storage with processed data
        for _, row in df_processed.iterrows():
            feedback_id = row['id']
            for feedback in feedback_storage:
                if feedback['id'] == feedback_id:
                    feedback['processed_text'] = row['feedback_text']
                    feedback['original_text'] = row['original_text']
                    feedback['features'] = {
                        'char_count': row.get('char_count', 0),
                        'word_count': row.get('word_count', 0),
                        'sentence_count': row.get('sentence_count', 0),
                        'exclamation_count': row.get('exclamation_count', 0),
                        'question_count': row.get('question_count', 0)
                    }
                    feedback['status'] = 'processed'
                    break
        
        # Generate summary
        summary = preprocessor.get_preprocessing_summary(df, df_processed)
        
        logger.info(f"Preprocessed {len(feedback_to_process)} feedback entries")
        
        return jsonify({
            'message': 'Preprocessing completed successfully',
            'processed_count': len(feedback_to_process),
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error preprocessing feedback: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/feedback/export', methods=['GET'])
def export_feedback():
    """
    Export feedback data in various formats.
    
    Query parameters:
    - format: Export format (json, csv) - default: json
    - include_processed: Include processed text (true/false) - default: false
    
    Returns:
        File download or JSON response
    """
    try:
        export_format = request.args.get('format', 'json').lower()
        include_processed = request.args.get('include_processed', 'false').lower() == 'true'
        
        if not feedback_storage:
            return jsonify({'error': 'No feedback data to export'}), 404
        
        # Prepare data for export
        export_data = []
        for feedback in feedback_storage:
            export_item = {
                'id': feedback['id'],
                'feedback_text': feedback['feedback_text'],
                'rating': feedback.get('rating'),
                'customer_id': feedback.get('customer_id'),
                'product_id': feedback.get('product_id'),
                'timestamp': feedback['timestamp'],
                'status': feedback['status']
            }
            
            if include_processed and 'processed_text' in feedback:
                export_item['processed_text'] = feedback['processed_text']
                export_item['features'] = feedback.get('features', {})
            
            export_data.append(export_item)
        
        if export_format == 'csv':
            df = pd.DataFrame(export_data)
            csv_data = df.to_csv(index=False)
            
            response = make_response(csv_data)
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = f'attachment; filename=feedback_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            
            return response
        
        else:  # JSON format
            return jsonify({
                'export_timestamp': datetime.now().isoformat(),
                'total_records': len(export_data),
                'data': export_data
            })
        
    except Exception as e:
        logger.error(f"Error exporting feedback: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/stats', methods=['GET'])
def get_statistics():
    """
    Get system statistics and analytics.
    
    Returns:
        JSON response with system statistics
    """
    try:
        if not feedback_storage:
            return jsonify({
                'total_feedback': 0,
                'message': 'No feedback data available'
            })
        
        # Basic statistics
        total_feedback = len(feedback_storage)
        status_counts = {}
        rating_counts = {}
        
        for feedback in feedback_storage:
            # Status distribution
            status = feedback.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Rating distribution
            rating = feedback.get('rating')
            if rating is not None:
                rating_counts[str(rating)] = rating_counts.get(str(rating), 0) + 1
        
        # Calculate averages
        ratings = [f.get('rating') for f in feedback_storage if f.get('rating') is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        
        # Text length statistics
        text_lengths = [len(f['feedback_text']) for f in feedback_storage]
        avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        
        stats = {
            'total_feedback': total_feedback,
            'status_distribution': status_counts,
            'rating_distribution': rating_counts,
            'average_rating': avg_rating,
            'average_text_length': avg_text_length,
            'latest_submission': max([f['timestamp'] for f in feedback_storage]) if feedback_storage else None
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({'error': 'Method not allowed'}), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Create some sample data for testing
    sample_feedback = [
        {
            'id': 1,
            'feedback_text': 'Great product! I love the quality and design.',
            'rating': 5,
            'customer_id': 'customer_001',
            'product_id': 'product_123',
            'metadata': {'source': 'website'},
            'timestamp': datetime.now().isoformat(),
            'status': 'submitted'
        },
        {
            'id': 2,
            'feedback_text': 'The service was terrible. Very disappointed.',
            'rating': 1,
            'customer_id': 'customer_002',
            'product_id': 'product_123',
            'metadata': {'source': 'email'},
            'timestamp': datetime.now().isoformat(),
            'status': 'submitted'
        }
    ]
    
    feedback_storage.extend(sample_feedback)
    
    logger.info("Starting AI-Powered Customer Feedback Analysis API")
    logger.info("Sample data loaded for testing")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

