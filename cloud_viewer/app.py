"""
Cloud-Ready Khipu 3D Viewer - Backend API

Modern Flask-based API server that serves khipu data from blob storage.
Supports Azure Blob Storage, AWS S3, or local filesystem.

Environment Variables:
    STORAGE_TYPE: 'local', 'azure', or 'aws' (default: local)
    STORAGE_PATH: Local path or container/bucket name
    AZURE_CONNECTION_STRING: Azure Storage connection string (if using Azure)
    AWS_ACCESS_KEY_ID: AWS access key (if using AWS)
    AWS_SECRET_ACCESS_KEY: AWS secret key (if using AWS)
    AWS_REGION: AWS region (if using AWS)
"""

import os
import json
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable CORS for API access

# Storage configuration
STORAGE_TYPE = os.environ.get('STORAGE_TYPE', 'local')
STORAGE_PATH = os.environ.get('STORAGE_PATH', '../data/blob_export')


class StorageBackend:
    """Abstract storage backend for khipu data."""
    
    def get_file(self, path: str) -> dict:
        """Retrieve and parse JSON file from storage."""
        raise NotImplementedError
    
    def list_khipus(self) -> list:
        """List all available khipus."""
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
    
    def get_file(self, path: str) -> dict:
        """Read JSON file from local filesystem."""
        file_path = self.base_path / path
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_khipus(self) -> list:
        """List all khipu JSON files."""
        khipus_dir = self.base_path / 'khipus'
        if not khipus_dir.exists():
            return []
        
        return [f.stem for f in khipus_dir.glob('*.json')]


class AzureBlobStorageBackend(StorageBackend):
    """Azure Blob Storage backend."""
    
    def __init__(self, container_name: str, connection_string: str):
        from azure.storage.blob import BlobServiceClient
        
        self.container_name = container_name
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service.get_container_client(container_name)
    
    def get_file(self, path: str) -> dict:
        """Download and parse JSON from Azure Blob Storage."""
        try:
            blob_client = self.container_client.get_blob_client(path)
            blob_data = blob_client.download_blob().readall()
            return json.loads(blob_data.decode('utf-8'))
        except Exception as e:
            print(f"Error reading blob {path}: {e}")
            return None
    
    def list_khipus(self) -> list:
        """List all khipu blobs."""
        blobs = self.container_client.list_blobs(name_starts_with='khipus/')
        return [blob.name.replace('khipus/', '').replace('.json', '') for blob in blobs]


class AWSStorageBackend(StorageBackend):
    """AWS S3 storage backend."""
    
    def __init__(self, bucket_name: str):
        import boto3
        
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
    
    def get_file(self, path: str) -> dict:
        """Download and parse JSON from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=path)
            data = response['Body'].read()
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            print(f"Error reading S3 object {path}: {e}")
            return None
    
    def list_khipus(self) -> list:
        """List all khipu objects in S3."""
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix='khipus/')
        if 'Contents' not in response:
            return []
        
        return [obj['Key'].replace('khipus/', '').replace('.json', '') 
                for obj in response['Contents']]


# Initialize storage backend
def get_storage_backend() -> StorageBackend:
    """Initialize appropriate storage backend based on configuration."""
    if STORAGE_TYPE == 'azure':
        connection_string = os.environ.get('AZURE_CONNECTION_STRING')
        if not connection_string:
            raise ValueError("AZURE_CONNECTION_STRING environment variable required for Azure storage")
        return AzureBlobStorageBackend(STORAGE_PATH, connection_string)
    
    elif STORAGE_TYPE == 'aws':
        return AWSStorageBackend(STORAGE_PATH)
    
    else:  # local
        return LocalStorageBackend(STORAGE_PATH)


storage = get_storage_backend()


# API Routes
@app.route('/')
def index():
    """Serve the main viewer page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'storage_type': STORAGE_TYPE,
        'storage_path': STORAGE_PATH
    })


@app.route('/api/colors')
def get_colors():
    """Get color mappings."""
    colors = storage.get_file('colors.json')
    if colors is None:
        return jsonify({'error': 'Colors not found'}), 404
    
    return jsonify(colors)


@app.route('/api/khipus')
def list_khipus():
    """List all available khipus."""
    index = storage.get_file('khipu_index.json')
    if index is None:
        return jsonify({'error': 'Khipu index not found'}), 404
    
    # Apply filters if provided
    provenance = request.args.get('provenance')
    min_cords = request.args.get('min_cords', type=int)
    max_cords = request.args.get('max_cords', type=int)
    
    filtered = index
    if provenance:
        filtered = [k for k in filtered if k.get('provenance') == provenance]
    if min_cords:
        filtered = [k for k in filtered if k.get('cord_count', 0) >= min_cords]
    if max_cords:
        filtered = [k for k in filtered if k.get('cord_count', 0) <= max_cords]
    
    return jsonify(filtered)


@app.route('/api/khipus/<khipu_id>')
def get_khipu(khipu_id):
    """Get detailed data for a specific khipu."""
    khipu_data = storage.get_file(f'khipus/{khipu_id}.json')
    if khipu_data is None:
        return jsonify({'error': 'Khipu not found'}), 404
    
    return jsonify(khipu_data)


@app.route('/api/stats')
def get_stats():
    """Get overall statistics."""
    index = storage.get_file('khipu_index.json')
    if index is None:
        return jsonify({'error': 'Khipu index not found'}), 404
    
    provenances = set(k.get('provenance', 'Unknown') for k in index)
    total_cords = sum(k.get('cord_count', 0) for k in index)
    total_knots = sum(k.get('knot_count', 0) for k in index)
    
    return jsonify({
        'total_khipus': len(index),
        'total_cords': total_cords,
        'total_knots': total_knots,
        'provenances': sorted(provenances),
        'avg_cords_per_khipu': round(total_cords / len(index), 1) if index else 0
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"🚀 Starting Khipu Cloud Viewer on port {port}")
    print(f"📦 Storage type: {STORAGE_TYPE}")
    print(f"📁 Storage path: {STORAGE_PATH}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
